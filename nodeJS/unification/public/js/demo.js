function demo_explain (thisElement){
    console.log(thisElement.alt);

    // Counter for the explain() loop - maximum 8
    var counter = 0;

    let imgName = thisElement.alt;
    console.log(imgName);
    let url = "";

    url += "/explanation-explain?";
    url += "dataset=Gun%20Wielding%20Image%20Classification";
    url += "&model=cnn_1";
    url += "&image=" + imgName;
    url += "&explanation=LIME";

    console.log(url);

    // Call explain API 9 times for selected image
    for (i=0; i < 9; i++) {
        console.log(i);
        let xmlHttp = new XMLHttpRequest();

        xmlHttp.onreadystatechange = function() {
            if (xmlHttp.readyState == 4) {
                if (xmlHttp.status == 200) {
                    let jsExp = JSON.parse(xmlHttp.responseText);
                    console.log(jsExp);

                    var resultDiv = "result" + counter;
                    console.log(resultDiv);
    
                    let matrixDiv = document.getElementById(resultDiv);
    
                    matrixDiv.innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation_image + "'>";
                    counter = counter +1;
                } else {
                    alert("The explanation failed - see server logs for details");
                }
            }
        }
    
        xmlHttp.open("GET", url, true);
        xmlHttp.send(null);
    }
}