const express = require('express');
const path = require("path")
const multer = require("multer");
const { threadId } = require('worker_threads');
const PythonShell = require('python-shell');
const app = express();
const fs = require('fs');

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
  fs.mkdir('./test_data', function(err){
    if(err){
      console.log(err)
    }else{
      console.log('created test_data')
    }
  })
  fs.mkdir('./results', function(err){
    if(err){
      console.log(err)
    }else{
      console.log('created results')
    }
  })
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
            }
            fs.rm('test_data', { recursive: true }, (err) => {
              if (err) {
                console.log(err);
              }
          
              console.log(`test_data is deleted!`);
            });
            fs.rm('results', { recursive: true }, (err) => {
              if (err) {
                console.log(err);
              }
          
              console.log(`results is cleared!`);
            });
          });
        }
    })
})

//fires if nothing yet has worked, needs to be at the end
