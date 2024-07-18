var option1 = document.getElementById("option1");
var option2 = document.getElementById("option2");
var recordButton = document.querySelector(".Record");
var stopButton = document.querySelector(".Stop");

option1.addEventListener("change", function() {
    if (option1.checked) {
        recordButton.disabled = true;
        stopButton.disabled = true;
    } else {
        recordButton.disabled = false;
        stopButton.disabled = false;
    }
});

option2.addEventListener("change", function() {
    if (option2.checked) {
        recordButton.disabled = false;
        stopButton.disabled = false;
    }
});