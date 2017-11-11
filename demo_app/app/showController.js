var child_process = require('child_process');
var resp;
var catcher = function(error, stdout, stderr) {
  if (error) {
    return resp.render('robo_bard', {content: stdout});
  }
  if (stdout) {
    return resp.render('robo_bard', {content: stdout});
  } else if (stderr) {
    return resp.render('robo_bard', {content: stdout});
  }
}

module.exports = {
  show: function (request, response) {
   return response.render('robo_bard');
  },
  tragedy: function(request, response) {
   resp = response;
   child_process.execFile('python',['./ml_script/gen_sent.py', 'tragedy'], {}, catcher);
  },
  comedy: function(request, response) {
   resp = response;
   child_process.execFile('python',['./ml_script/gen_sent.py', 'comedy'], {}, catcher);
  },
  history: function(request, response) {
   resp = response;
   child_process.execFile('python',['./ml_script/gen_sent.py', 'history'], {}, catcher);
  }
}
