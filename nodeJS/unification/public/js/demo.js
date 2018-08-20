function demo_explain (thisElement){
    console.log(thisElement.alt);

    let imgName = thisElement.alt;
    console.log(imgName);
    let xmlHttp = new XMLHttpRequest();
    let url = "";

    url += "/explanation-explain?";
    url += "dataset=Gun%20Wielding%20Image%20Classification";
    url += "&model=cnn_1";
    url += "&image=" + imgName;
    url += "&explanation=LIME";

    console.log(url);

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4) {
            console.log("hihihi");
            if (xmlHttp.status == 200) {
                console.log("hihihi");
                let jsExp = JSON.parse(xmlHttp.responseText);
                console.log(jsExp);

                let matrix1 = document.getElementById("result1");

                matrix1.innerHTML = "<img alt='Explanation image' src='data:img/jpg;base64," + jsExp.explanation_image + "'>";
            } else {
                alert("The explanation failed - see server logs for details");
            }
        }
    }

    xmlHttp.open("GET", url, true);
    xmlHttp.send(null);
}