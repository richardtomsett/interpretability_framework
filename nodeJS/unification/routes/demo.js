// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');

/* GET api-tester page. */
router.get('/', function(req, res, next) {
    res.render("demo", {
        "title": "P5 demo - AFM Demo"
    });
});

module.exports = router;
