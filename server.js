const process = require("./pix2pix.js");
const express = require("express");
const bodyParser = require("body-parser");
const imageToBase64 = require("image-to-base64");
const multer = require("multer");
var storage = multer.diskStorage({
    destination: function(req, file, cb) {
        cb(null, "");
    },
    filename: function(req, file, cb) {
        cb(null, "process.png");
    },
});
const app = express();
app.use(express.static("public"));
app.use(
    bodyParser.urlencoded({
        extended: true,
    })
);
app.get("/", function(req, res) {
    res.send({ status: "ok" });
});
var upload = multer({
    storage: storage,
});
app.post("/upload", upload.single("myFile"), async(req, res, next) => {
    const file = req.file;
    if (!file) {
        const error = new Error("Please upload a file");
        error.httpStatusCode = 400;
        return next(error);
    }
    let base64 = await imageToBase64("./" + file.path.toString());
    process(res);
});
app.post("/uploadFile", upload.single("myFile"), async(req, res, next) => {
    const file = req.file;
    if (!file) {
        const error = new Error("Please upload a file");
        error.httpStatusCode = 400;
        return next(error);
    }
    res.sendFile(__dirname + "/result.png");
});
app.listen(3000, function() {
    console.log("Server is running on port 3000");
});