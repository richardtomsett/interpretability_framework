// *******************************************************************************
// * (C) Copyright IBM Corporation 2018
// * All Rights Reserved
// *******************************************************************************

function useSelectedDataset() {
    let modUrl = "/models-for-dataset?dataset=" + getSelectedDatasetName();
    let imgUrl = "/dataset-details?dataset=" + getSelectedDatasetName();

    populateModels(modUrl);
    populateStaticImages(imgUrl);
}

function populateExplanations() {
    let url = "";
    let xmlHttp = new XMLHttpRequest();

    url += "/explanations-for-filter?";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsExps = JSON.parse(xmlHttp.responseText);

                let eXl = document.getElementById("exp_list");
                clearAllOptions(eXl);

                for (let i in jsExps.explanations) {
                    let thisExp = jsExps.explanations[i];
                    let option = document.createElement("option");
                    option.text = thisExp.explanation_name;
                    eXl.add(option);
                }
            } else {
                alert("The list of explanation types failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function populateModels(url) {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsMods = JSON.parse(xmlHttp.responseText);

                let eSm = document.getElementById("settings_model");
                let eMl = document.getElementById("mod_list");
                clearAllOptions(eMl);

                eSm.style.display = "block";

                for (let i in jsMods.models) {
                    let thisMod = jsMods.models[i];
                    let option = document.createElement("option");
                    option.text = thisMod.model_name;
                    eMl.add(option);
                }
            } else {
                alert("The list of models failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function populateStaticImages(url) {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsDs = JSON.parse(xmlHttp.responseText);
                let eIin = document.getElementById("img_input_name");
                let eIl = document.getElementById("img_list");
                clearAllOptions(eIl);

                for (let i in jsDs.interesting_images) {
                    let option = document.createElement("option");
                    option.text = jsDs.interesting_images[i];
                    eIl.add(option);
                }

                eIin.value = jsDs.interesting_images[0];
            } else {
                alert("The list of static images failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function clearAllOptions(e) {
    for(let i = e.options.length - 1; i >= 0; i--)
    {
        e.remove(i);
    }
}

function useSelectedModel() {
    let eSi = document.getElementById("settings_image");

    eSi.style.display = "block";

    populateExplanations();
}

function useRandomImage() {
    let url = "/dataset-test-image?dataset=" + getSelectedDatasetName();

    requestImage(url);
}

function showRandomImageBatch() {
    let url = "";
    let e = document.getElementById("img_num_images");
    let num_images = e.value;

    url += "/dataset-test-images";
    url += "?dataset=" + getSelectedDatasetName();
    url += "&num_images=" + num_images;

    requestImageBatch(url);
}

function usePredefinedImage(imageName) {
    if (imageName == null) {
        imageName = getPredefinedImageName();
    }

    let url = "/dataset-test-image?dataset=" + getSelectedDatasetName() + "&image=" + imageName;

    requestImage(url);
}

function getSelectedDatasetName() {
    let e = document.getElementById("ds_list");

    return e.options[e.selectedIndex].value;
}

function getSelectedModelName() {
    let e = document.getElementById("mod_list");

    return e.options[e.selectedIndex].value;
}

function getSelectedExplanationName() {
    let e = document.getElementById("exp_list");

    return e.options[e.selectedIndex].value;
}

function getSelectedImageName() {
    let e = document.getElementById("img_name");

    return e.innerHTML;
}

function getPredefinedImageName() {
    let e = document.getElementById("img_list");

    return e.options[e.selectedIndex].value;
}

function useNamedImage() {
    let e = document.getElementById("img_input_name");
    let imageName = e.value;

    usePredefinedImage(imageName);
}

function requestImage(url) {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsImg = JSON.parse(xmlHttp.responseText);

                let eDi = document.getElementById("div_image");
                let eIn = document.getElementById("img_name");
                let eIi = document.getElementById("img_image");
                let eIg = document.getElementById("img_groundtruth");
                let eDe = document.getElementById("div_exps");

                eDi.style.display = "block";
                eIn.innerHTML = jsImg.image_name;
                eIi.innerHTML = "<img alt='" + jsImg.image_name + "' src='data:img/jpg;base64," + jsImg.input + "'/>";
                eIg.innerHTML = jsImg.ground_truth;
                eDe.style.display = "block";
            } else {
                alert("The image failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function requestImageBatch(url) {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsImgList = JSON.parse(xmlHttp.responseText);
                let numImages = jsImgList.length;
                console.log(jsImgList);

                let eIl = document.getElementById("div_image_list");
                let eR01 = document.getElementById("row_img_num");
                let eR02 = document.getElementById("row_img_name");
                let eR03 = document.getElementById("row_img_img");
                let eR04 = document.getElementById("row_img_gt");
                let eR05 = document.getElementById("row_img_act");

                for (let i in jsImgList) {
                    let thisImg = jsImgList[i];
                    let cell01 = eR01.insertCell(-1);
                    let cell02 = eR02.insertCell(-1);
                    let cell03 = eR03.insertCell(-1);
                    let cell04 = eR04.insertCell(-1);
                    let cell05 = eR05.insertCell(-1);

                    let htmlPredict = "<button id='pred_button', class=\"btn btn-primary my-2\", onclick=\"javascript:predictFor('" + thisImg.image_name + "');\">Predict</button>";

                    cell01.innerHTML = (parseInt(i) + 1) + " of " + numImages;
                    cell02.innerHTML = thisImg.image_name;
                    cell03.innerHTML = "<img alt='" + thisImg.image_name + "' src='data:img/jpg;base64," + thisImg.input + "'/>";
                    cell04.innerHTML = thisImg.ground_truth;
                    cell05.innerHTML = htmlPredict;
                }

                eIl.style.display = "block";
            } else {
                alert("The image batch failed to be retrieved - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function explain() {
    let xmlHttp = new XMLHttpRequest();
    let tgtMsg = document.getElementById("in_progress");
    let tgtBut = document.getElementById("exp_button");
    let tgtTbl = document.getElementById("exp_table");
    let ts = Date.now();
    let url = "";

    url += "/explanation-explain?";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();
    url += "&explanation=" + getSelectedExplanationName();
    url += "&image=" + getSelectedImageName() ;

    tgtMsg.style.display = "block";
    tgtBut.style.display = "none";

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsExp = JSON.parse(xmlHttp.responseText);
                console.log(jsExp);

                let row = tgtTbl.insertRow(1);
                let cell1 = row.insertCell(0);
                let cell2 = row.insertCell(1);
                let cell3 = row.insertCell(2);
                let cell4 = row.insertCell(3);
                let cell5 = row.insertCell(4);
                let cell6 = row.insertCell(5);
                let cell7 = row.insertCell(6);
                let cell8 = row.insertCell(7);
                let cell9 = row.insertCell(8);
                let cell10 = row.insertCell(9);
                let cell11 = row.insertCell(10); // dhm

                cell1.innerHTML = "";
                cell1.innerHTML += "Dataset=" + getSelectedDatasetName() + "<br/>";
                cell1.innerHTML += "Model=" + getSelectedModelName() + "<br/>";
                cell1.innerHTML += "Explanation=" + getSelectedExplanationName() + "<br/>";

                cell2.innerHTML = formattedDateTime();
                cell3.innerHTML = Date.now() - ts;
                cell4.innerHTML = jsExp.prediction;
                cell5.innerHTML = jsExp.explanation_text;
                cell6.innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation_image + "'>";
                cell7.innerHTML = "<img alt='Boundary image' src='data:img/jpg;base64," + jsExp.boundary_image + "'>";
                cell8.innerHTML = "<img alt='Average Explanation image' src='data:img/jpg;base64," + jsExp.average_picture + "'>";
                cell9.innerHTML = "<img alt='Three Region image' src='data:img/jpg;base64," + jsExp.three_region_picture + "'>";
                cell10.innerHTML = "<img alt='Standard Deviation image' src='data:img/jpg;base64," + jsExp.standard_deviation_picture + "'>";
                cell11.innerHTML = "<img alt='Tinted image' src='data:img/jpg;base64," + jsExp.tinted_image + "'>"; // dhm

                tgtMsg.style.display = "none";
                tgtBut.style.display = "block";
            } else {
                tgtMsg.style.display = "none";
                tgtBut.style.display = "block";

                alert("The explanation failed - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function predict() {
    let xmlHttp = new XMLHttpRequest();
    let tgtMsg = document.getElementById("pred_result");
    let url = "";

    url += "/model-predict?";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();
    url += "&image=" + getSelectedImageName() ;

    tgtMsg.innerHTML = "Prediction in progress...";

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsPred = JSON.parse(xmlHttp.responseText);

                tgtMsg.innerHTML = jsPred.prediction;
            } else {
                alert("The predict request failed - see server logs for details");
                tgtMsg.innerHTML = "Prediction failed";
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function predictFor(imgName, tgtElemName) {
    let xmlHttp = new XMLHttpRequest();
    let tgtMsg = document.getElementById(tgtElemName);
    let url = "";

    url += "/model-predict?";
    url += "&dataset=" + getSelectedDatasetName();
    url += "&model=" + getSelectedModelName();
    url += "&image=" + imgName;

    tgtMsg.innerHTML = "Prediction in progress...";

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            if (xmlHttp.status == 200) {
                let jsPred = JSON.parse(xmlHttp.responseText);

                tgtMsg.innerHTML = jsPred.prediction;
            } else {
                alert("The predict request failed - see server logs for details");
                tgtMsg.innerHTML = "Prediction failed";
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}

function formattedDateTime() {
    var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    var today = new Date();
    var day = today.getDate();

    var mon = today.getMonth();
    var year = today.getFullYear();
    var hour = today.getHours();
    var min = today.getMinutes();
    var sec = today.getSeconds();

    if (day < 10) {
        day = "0" + day;
    }

    return day + "-" + months[mon] + "-" + year + " " + hour + ":" + min + ":" + sec;
}