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
        loadingDiv.innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";
        document.getElementById("result_average").innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";
        document.getElementById("result_significant").innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";
        document.getElementById("result_tinted").innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";
        document.getElementById("result_sd").innerHTML = "<img alt='Explanation image' src='../img/loading.gif'>";


        xmlHttp.onreadystatechange = function() {
            if (xmlHttp.readyState == 4) {
                if (xmlHttp.status == 200) {
                    let jsExp = JSON.parse(xmlHttp.responseText);
                    console.log(jsExp);

                    var resultDiv = "result" + counter;
                    let matrixDiv = document.getElementById(resultDiv);
    
                    matrixDiv.innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation_image + "'>";
                    counter = counter +1;

                    // For each iteration, display the compounded averaged images
                    // if (counter == 9) {
                        document.getElementById("result_average").innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation_image + "'>";
                        document.getElementById("result_significant").innerHTML = "<img alt='Average Explanation image' src='data:img/jpg;base64," + jsExp.average_picture + "'>";
                        document.getElementById("result_tinted").innerHTML = "<img alt='Tinted image' src='data:img/jpg;base64," + jsExp.tinted_image + "'>";
                        document.getElementById("result_sd").innerHTML = "<img alt='Standard Deviation image' src='data:img/jpg;base64," + jsExp.standard_deviation_picture + "'>";
                    // }
                    
                } else {
                    alert("The explanation failed - see server logs for details");
                }
            }
        }
    
        xmlHttp.open("GET", url, true);
        xmlHttp.send(null);
    }
}