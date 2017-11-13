var child_process = require('child_process');
var resp;
var catcher = function(error, stdout, stderr) {
  if (error) {
    resp.render('robo_bard', {content: error});
  }
  if (stdout) {
    resp.render('robo_bard', {content: stdout});
  } else if (stderr) {
    resp.render('robo_bard', {content: stderr});
  }

  process.chdir('./demo_app');
}

var launchGenSent = function(type, response) {
  resp = response;
  process.chdir('..');
  child_process.execFile('python',['gen_sent.py', type], {}, catcher);
}

module.exports = {
  show: function (request, response) {
    return response.render('robo_bard');
  },
  tragedy: function(request, response) {
    launchGenSent('tragedy', response);
  },
  comedy: function(request, response) {
    launchGenSent('comedy', response);
  },
  history: function(request, response) {
    launchGenSent('history', response);
  }
}
