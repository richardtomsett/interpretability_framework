// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

let express = require('express');
let router = express.Router();
let request = require('request-promise');
let config = require('../config');
let fn = require('./functions-general');
let parmType = null;
let parmDs = null;

router.get('/', function (req, res) {
    parmType = req.query.type;
    parmDs = req.query.dataset;

    if (parmDs != null) {
        const options = {
            method: 'GET',
            uri: fn.getDatasetsAllUrl(config)
        };

        request(options)
            .then(function (response) {
                // Success
                let result = JSON.parse(response);
                let matchedDs = fn.matchedDataset(parmDs, result.datasets);

                if (matchedDs == null) {
                    let errMsg = "Error - no dataset matches '" + parmDs + "'";
                    return res.status(500).send(errMsg);
                } else {
                    if (parmType == "html") {
                        let jsPage = {
                            "title": config.unified_apis.dataset.details.url,
                            "dataset": matchedDs,
                            "parameters": {
                                "type": parmType,
                                "dataset": parmDs
                            }
                        };

                        res.render(config.unified_apis.dataset.details.route, jsPage);
                    } else {
                        res.json(matchedDs);
                    }
                }
            })
            .catch(function (err) {
                // Error
                console.log(err);
                return res.sendStatus(500);
            })
    } else {
        let errMsg = "Error: No dataset specified";
        return res.status(500).send(errMsg);
    }
});

module.exports = router;
