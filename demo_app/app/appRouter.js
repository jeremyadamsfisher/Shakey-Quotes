var showController = require('./showController.js');
module.exports = function(express) {
    var router = express.Router();
    router.get('/', showController.show);
    router.get('/tragedy', showController.tragedy);
    router.get('/comedy', showController.comedy);
    return router;
};
