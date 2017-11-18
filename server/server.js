var express = require('express');
var bodyParser = require('body-parser');
var app = express();

//Allow all requests from all domains & localhost
app.all('/*', function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "X-Requested-With, Content-Type, Accept");
  res.header("Access-Control-Allow-Methods", "POST, GET");
  next();
});

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: false}));
app.use(express.static('web'));


var rooms = {
    1: {},
    2: {}
};

app.get('/rooms', function(req, res) {
    console.log("GET TO ROOMS");
    res.header("Content-Type", "application/json")
    res.send(Object.keys(rooms));
});

app.get('/rooms/all', function(req, res) {
    console.log("GET TO ROOMS ALL")
    res.header("Content-Type", "application/json")
    res.send(rooms);
});

app.post('/rooms', function(req, res) {
    console.log("POST TO ROOMS");
    var id = new Date().getUTCMilliseconds();
    rooms[id] = [];
    res.header("Content-Type", "application/json")
    res.status(201).send(JSON.stringify({id: id}));
});

app.get('/rooms/:roomId/events', function(req, res) {
    console.log("GET TO EVENTS");
    var id = req.params.roomId;
    res.send(rooms[id]);
});

app.post('/rooms/:roomId/events', function(req, res) {
    console.log("POST TO EVENTS");
    var events = req.body;
    
    for (var key in events) {
        rooms[req.params.roomId][key] = events[key];
    }
    
    res.header("Content-Type", "application/json")
    res.status(201).send(req.body);
});

app.listen(8080);


