const path = require('path');
const express = require('express');
const appRouter = require('./app/appRouter.js')(express);
const exphbs = require('express-handlebars');
const port = process.env.PORT || 8080;
const app = express();


app.engine('.hbs', exphbs({
  defaultLayout: 'main',
  extname: '.hbs',
  layoutsDir: path.join(__dirname, 'app/views/layouts')
}));


app.set('view engine', '.hbs');
app.set('views', path.join(__dirname, 'app/views'));

var bodyParser = require('body-parser');

app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('app/static'));

app.use((err, request, response, next) => {
  // log the error, for now just console.log
  console.log(err);
  response.status(500).send('Something broke!');
});

app.use('/', appRouter);

app.listen(port, (err) => {
  if (err) {
    return console.log('something bad happened', err)
  }

  console.log(`server is listening on ${port}`);
});

module.exports.getApp = app;

