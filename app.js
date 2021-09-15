const express = require('express');
const path = require("path")
const multer = require("multer");
const { threadId } = require('worker_threads');
const PythonShell = require('python-shell');
const app = express();
var fs = require('fs');

//register view engine
app.set('view engine', 'ejs');


//listen to requests
app.listen(3000);

var storage = multer.diskStorage({
  destination: function (req, file, cb) {

      // Uploads is the Upload_folder_name
      cb(null, "test_data")
  },
  filename: function (req, file, cb) {
    cb(null, file.fieldname + "-" + Date.now()+".jpg")
  }
})

const maxSize = 1 * 1000 * 1000;
    
var upload = multer({ 
    storage: storage,
    limits: { fileSize: maxSize },
    fileFilter: function (req, file, cb){
    
        // Set the filetypes, it is optional
        var filetypes = /jpeg|jpg|png/;
        var mimetype = filetypes.test(file.mimetype);
  
        var extname = filetypes.test(path.extname(
                    file.originalname).toLowerCase());
        
        if (mimetype && extname) {
            return cb(null, true);
        }
      
        cb("Error: File upload only supports the "
                + "following filetypes - " + filetypes);
      } 
  
// mypic is the name of file attribute
}).single("mypic");       
  
app.get("/",function(req,res){
    res.render("upload");
})
    
app.post("/uploadProfilePicture", async function (req, res, next) {
        
    // Error MiddleWare for multer file upload, so if any
    // error occurs, the image would not be uploaded!
    upload(req,res,async function(err) {
  
        if(err) {
  
            // ERROR occured (here it can be occured due
            // to uploading image of size greater than
            // 1MB or uploading different file type)
            res.send(err)
        }
        else {
  
            // SUCCESS, image successfully uploaded
            //res.send("Success, Image uploaded! Please wait around 30 seconds until we generate your photo!")
            const spawn = require("child_process").spawn;
            console.log('reached')
            const pythonProcess = spawn('python',["rotatedetect.py"]); 
            
            pythonProcess.stdout.on('data', (data) => {
              console.log(data.toString())
            if (data.toString() != "failed"){
              console.log("reached good ending");
              //res.send("Successfully created image")
              res.sendFile('/results/res.png', {root: __dirname});
              //res.sendFile('./views/index.html', {root: __dirname});
            }else{
              res.send("Couldn't detect face in image. Please try another photo.");
              // fs.unlink('mynewfile2.txt', function (err) {
              //   if (err) throw err;
              //   console.log('File deleted!');
              // });
            }
          });
        }
    })
})

app.get('/view_image', (req, res) => {
  console.log('/');
    //res.send('<p> home page </p>'); //automatically infers content type and status code
    res.render('index', {root: __dirname});
});

app.get('/', (req, res) => {
  console.log('/');
    //res.send('<p> home page </p>'); //automatically infers content type and status code
    res.sendFile('./views/upload.html', {root: __dirname});
});

app.post('/upload', async (req, res) => {
  console.log('upload');
  //res.sendFile('./views/upload.html', {root: __dirname});
  try {
      console.log(req.files)
      if(!req.files) {
          res.send({
              status: false,
              message: 'No file uploaded'
          });
      } else {
          //Use the name of the input field (i.e. "avatar") to retrieve the uploaded file
          let photo = req.files.filename;
          
          //Use the mv() method to place the file in upload directory (i.e. "uploads")
          photo.mv('./test_data/' + photo.name);

          //send response
          res.send({
              status: true,
              message: 'File is uploaded',
              data: {
                  name: photo.name,
                  mimetype: photo.mimetype,
                  size: photo.size
              }
          });
          console.log('we have reached somewhere');
          const spawn = require("child_process").spawn;
          const pythonProcess = spawn('python',["./rotatedetect.py"]);
          pythonProcess.stdout.on('data', (data) => {
            if (data == "success"){
              res.sendFile('./views/index.html', {root: __dirname});
            }else{
              res.write("Couldn't detect face in image. Please try another photo.");
            }
        });

      }
  } catch (err) {
      res.status(500).send(err);
  }
});

//404 page
app.use((req, res) => {
    res.sendFile('./views/404.html', {root: __dirname});
});
//fires if nothing yet has worked, needs to be at the end
