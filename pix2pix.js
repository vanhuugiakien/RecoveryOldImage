const fs = require("fs");
const dl = require("./deeplearn-0.3.15");
const PNG = require("png-js");
const PNG2 = require("pngjs2").PNG;
const sharp = require("sharp");
const imageToBase64 = require("image-to-base64");

let process = function(res) {
    console.log("start");
    fetchWeights("./model/model.h5").then(
        async(weights) => {
            // console.log(weights);
            const SIZE = 256;
            // var input_uint8_data = new Uint8ClampedArray();
            let buffer = fs.readFileSync("process.png");
            let imgResized = await sharp(buffer).resize(SIZE, SIZE).png().toBuffer();
            // console.log(buffer);

            fs.writeFileSync("process.png", new Uint8ClampedArray(imgResized));
            // console.log(path);
            const byteData = await new Promise((resolve, reject) => {
                PNG.decode("process.png", function(pixels) {
                    resolve(pixels);
                });
            });

            // console.log(input_uint8_data.length);
            var inputFloat32Data = Float32Array.from(byteData, (x) => x / 255);
            // console.log(inputFloat32Data);

            console.time("render");
            const math = dl.ENV.math;
            math.startScope();

            var inputRGBA = dl.Array3D.new(
                [SIZE, SIZE, 4],
                inputFloat32Data,
                "float32"
            );
            var input_rgb = math.slice3D(inputRGBA, [0, 0, 0], [SIZE, SIZE, 3]);
            console.log("processing ...");
            var outputRGB = model(input_rgb, weights);
            var alpha = dl.Array3D.ones([SIZE, SIZE, 1]);
            var outputRGBA = math.concat3D(outputRGB, alpha, 2);
            outputRGBA.getValuesAsync().then(async(output_float32_data) => {
                var outputUint8Data = Uint8ClampedArray.from(
                    output_float32_data,
                    (x) => x * 255
                );
                console.log(outputUint8Data);
                var imgPNG = new PNG2({ width: SIZE, height: SIZE });
                imgPNG.data = Buffer.from(outputUint8Data);
                imgPNG
                    .pack()
                    .pipe(fs.createWriteStream("result.png"))
                    .on("close", function() {
                        // let base64 = await imageToBase64("result.png");
                        res.sendFile(__dirname + "/result.png");
                    });
                math.endScope();
                console.timeEnd("render");
            });
        },
        (e) => {}
    );
};
module.exports = process;

function fetchWeights(path) {
    let weightsCache = {};
    return new Promise(function(resolve, reject) {
        if (path in weightsCache) {
            resolve(weightsCache[path]);
            return;
        }

        const buf = fs.readFileSync(path, null).buffer;

        var parts = [];
        var offset = 0;
        while (offset < buf.byteLength) {
            var b = new Uint8Array(buf.slice(offset, offset + 4));
            offset += 4;
            var len = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3];
            parts.push(buf.slice(offset, offset + len));
            offset += len;
        }
        var shapes = JSON.parse(new TextDecoder("utf8").decode(parts[0]));
        var index = new Float32Array(parts[1]);
        var encoded = new Uint8Array(parts[2]);
        var arr = new Float32Array(encoded.length);
        for (var i = 0; i < arr.length; i++) {
            arr[i] = index[encoded[i]];
        }

        var weights = {};
        var offset = 0;
        for (var i = 0; i < shapes.length; i++) {
            var shape = shapes[i].shape;
            var size = shape.reduce((total, num) => total * num);
            var values = arr.slice(offset, offset + size);
            var dlarr = dl.Array1D.new(values, "float32");
            weights[shapes[i].name] = dlarr.reshape(shape);
            offset += size;
        }
        weightsCache[path] = weights;
        resolve(weights);
    });
}

function model(input, weights) {
    const math = dl.ENV.math;

    function preprocess(input) {
        return math.subtract(
            math.multiply(input, dl.Scalar.new(2)),
            dl.Scalar.new(1)
        );
    }

    function deprocess(input) {
        return math.divide(math.add(input, dl.Scalar.new(1)), dl.Scalar.new(2));
    }

    function batchnorm(input, scale, offset) {
        var moments = math.moments(input, [0, 1]);
        const varianceEpsilon = 1e-5;
        return math.batchNormalization3D(
            input,
            moments.mean,
            moments.variance,
            varianceEpsilon,
            scale,
            offset
        );
    }

    function conv2d(input, filter, bias) {
        return math.conv2d(input, filter, bias, [2, 2], "same");
    }

    function deconv2d(input, filter, bias) {
        var convolved = math.conv2dTranspose(
            input,
            filter, [input.shape[0] * 2, input.shape[1] * 2, filter.shape[2]], [2, 2],
            "same"
        );
        var biased = math.add(convolved, bias);
        return biased;
    }
    var preprocessed_input = preprocess(input);
    var layers = [];
    var filter = weights["generator/encoder_1/conv2d/kernel"];
    var bias = weights["generator/encoder_1/conv2d/bias"];
    var convolved = conv2d(preprocessed_input, filter, bias);
    layers.push(convolved);
    for (var i = 2; i <= 8; i++) {
        var scope = "generator/encoder_" + i.toString();
        var filter = weights[scope + "/conv2d/kernel"];
        var bias = weights[scope + "/conv2d/bias"];
        var layer_input = layers[layers.length - 1];
        var rectified = math.leakyRelu(layer_input, 0.2);
        var convolved = conv2d(rectified, filter, bias);
        var scale = weights[scope + "/batch_normalization/gamma"];
        var offset = weights[scope + "/batch_normalization/beta"];
        var normalized = batchnorm(convolved, scale, offset);
        layers.push(normalized);
    }

    for (var i = 8; i >= 2; i--) {
        if (i == 8) {
            var layer_input = layers[layers.length - 1];
        } else {
            var skip_layer = i - 1;
            var layer_input = math.concat3D(
                layers[layers.length - 1],
                layers[skip_layer],
                2
            );
        }
        var rectified = math.relu(layer_input);
        var scope = "generator/decoder_" + i.toString();
        var filter = weights[scope + "/conv2d_transpose/kernel"];
        var bias = weights[scope + "/conv2d_transpose/bias"];
        var convolved = deconv2d(rectified, filter, bias);
        var scale = weights[scope + "/batch_normalization/gamma"];
        var offset = weights[scope + "/batch_normalization/beta"];
        var normalized = batchnorm(convolved, scale, offset);
        // missing dropout
        layers.push(normalized);
    }
    var layer_input = math.concat3D(layers[layers.length - 1], layers[0], 2);
    var rectified = math.relu(layer_input);
    var filter = weights["generator/decoder_1/conv2d_transpose/kernel"];
    var bias = weights["generator/decoder_1/conv2d_transpose/bias"];
    var convolved = deconv2d(rectified, filter, bias);
    var rectified = math.tanh(convolved);
    layers.push(rectified);
    var output = layers[layers.length - 1];
    var deprocessed_output = deprocess(output);
    return deprocessed_output;
}