let file = null;
let modelId = null;

document.getElementById("fileUpload").addEventListener("change", async (e) => {
    file = e.target.files[0];
    const preview = document.getElementById("preview");
    const filePara = document.getElementById("filePath");
    const reader = new FileReader();
    if (file) {
        reader.onload = function (e) {
            preview.src = e.target.result;  
        };
        reader.readAsDataURL(file);
        
        preview.style.display = "block";
    }
    else{
        preview.style.display = "none";
    }
    filePara.innerText = document.getElementById("fileUpload").value.split("\\").pop();
    
});

document.getElementById("PredictForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    modelId = document.getElementById("modelSelect").value;
    
    if(!file){
        alert("Select an image to analyze");
        return;
    }
    if(!modelId){
        alert("Insert a model");
        return;
    }

    const data = new FormData();
    data.append("file", file);
    data.append("modelId", modelId);

    try{
        predResult = await fetch("http://127.0.0.1:5000/predict", {
            method:"post",
            body : data
        });
        result = await predResult.json();
        document.getElementById("resultText").innerText = predResult;
        if(result["error"]){
            alert(`ERROR ${result.error}`);
        }
        else{
            document.getElementById("resultText").innerText = result["prediction"];
        }
    }
    catch (e){
        console.error("ERRORE: " + e);
        alert("Error while processing image");
        return
    }

})