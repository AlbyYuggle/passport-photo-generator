const express = require('express');

const app = express();

//register view engine
app.set('view engine', 'ejs');


//listen to requests
app.listen(3000);

app.get('/', (req, res) => {
    //res.send('<p> home page </p>'); //automatically infers content type and status code
    res.sendFile('./views/index.html', {root: __dirname});
});

//404 page
app.use((req, res) => {
    res.sendFile('./views/404.html', {root: __dirname});
});
//fires if nothing yet has worked, needs to be at the end