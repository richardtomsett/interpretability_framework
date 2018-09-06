function demo_explain (thisElement){

    // Counter for the explain() loop - maximum 8
    var counter = 0;

    // Extract image name from the alt
    let imgName = thisElement.alt;
    
    // Set up URL for selected image
    let url = "";
    url += "/explanation-explain?";
    url += "dataset=Gun%20Wielding%20Image%20Classification";
    url += "&model=cnn_1";
    url += "&image=" + imgName;
    url += "&explanation=LIME";
    console.log(url);

    // Call explain API 9 times for selected image
    for (i=0; i < 9; i++) {
        let xmlHttp = new XMLHttpRequest();

        // Set loading gif to boxes
        let loadingDiv = document.getElementById("result" + i);
        let tintloadingDiv = document.getElementById("tint_result" + i);
        tintloadingDiv.innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";
        loadingDiv.innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";
        document.getElementById("final_result_average").innerHTML = "<img id='loading1' alt='Explanation image' class='loadinggif' src='../img/loading.gif'>";
        document.getElementById("final_result_significant").innerHTML = "<img id='loading2' alt='Explanation image'class='loadinggif' src='../img/loading.gif'>";
        document.getElementById("final_result_sd").innerHTML = "<img id='loading3' alt='Explanation image' class='loadinggif' src='../img/loading.gif'>";


        xmlHttp.onreadystatechange = function() {
            if (xmlHttp.readyState == 4) {
                if (xmlHttp.status == 200) {
                    let jsExp = JSON.parse(xmlHttp.responseText);
                    console.log(jsExp);

                    var resultDiv = "result" + counter;
                    var tintresultDiv = "tint_result" + counter;
                    let matrixDiv = document.getElementById(resultDiv);
                    let tintmatrixDiv = document.getElementById(tintresultDiv);
    
                    // Add Explanation image and tinted images to result matrix
                    matrixDiv.innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation_image +"'>";
                    tintmatrixDiv.innerHTML = "<img alt='Tinted Explanation image' src='data:img/jpg;base64," + jsExp.tinted_image + "'>";

                    // Single Average images - instantiate parent divs
                    var parent_average = document.getElementById("result_average");
                    var parent_significant = document.getElementById("result_significant");
                    var parent_sd = document.getElementById("result_sd");

                    // Instantiate child image elements
                    var img_average = document.createElement("IMG");
                    img_average.src = "data:img/jpg;base64," + jsExp.average_picture;
                    var img_significant = document.createElement("IMG");
                    img_significant.src = "data:img/jpg;base64," + jsExp.three_region_picture;
                    var img_sd = document.createElement("IMG");
                    img_sd.src = "data:img/jpg;base64," + jsExp.standard_deviation_picture;

                    // Add child image to parent div
                    parent_average.appendChild(img_average, parent_average.childNodes[0]);
                    parent_significant.appendChild(img_significant, parent_significant.childNodes[0]);
                    parent_sd.appendChild(img_sd, parent_sd.childNodes[0]);

                    counter = counter +1;

                    if (counter == 8) {
                        document.getElementById("final_result_average").innerHTML = "<img alt='Result image' src='data:img/jpg;base64," + jsExp.average_picture +"'>";
                        document.getElementById("final_result_significant").innerHTML = "<img alt='Result image' src='data:img/jpg;base64," + jsExp.three_region_picture +"'>";
                        document.getElementById("final_result_sd").innerHTML = "<img alt='Result image' src='data:img/jpg;base64," + jsExp.standard_deviation_picture +"'>";
                    }
                    
                } else {
                    alert("The explanation failed - see server logs for details");
                }
            }
        }
    
        xmlHttp.open("GET", url, true);
        xmlHttp.send(null);
    }
}