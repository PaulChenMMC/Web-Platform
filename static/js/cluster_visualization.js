//回傳選擇測試類別
function sendSelectedRecordType(recordType) {
    var formData = new FormData();
    formData.append('Record_type', recordType);
    var request = new XMLHttpRequest();
    request.open('POST', '/get_folders');
    request.onload = function() {
        if (request.status === 200) {
            var folders = JSON.parse(request.responseText);
            updateSelectOptions(folders);
        } else {
            console.error('Failed to get folders');
        }
    };
    request.onerror = function() {
        console.error('Failed to get folders');
    };
    request.send(formData);
}

//回傳資料夾
function updateSelectOptions(folders) {
    var selectIDSelect = document.getElementById('Select_ID');
    selectIDSelect.innerHTML = ''; 

    folders.forEach(function(folder) {
        var option = document.createElement('option');
        option.text = folder;
        option.value = folder;
        selectIDSelect.appendChild(option);
    });
}

//日期輸入器
document.addEventListener('DOMContentLoaded', function() {
    var startDatePicker = document.getElementById('start-datePicker');

    startDatePicker.addEventListener('input', function() {
        var startDate = startDatePicker.value;
        endDatePicker.min = startDate;
    });

    const specialDates = ['2024-04-06', '2024-04-10', '2024-04-15'];
    if (specialDates.includes(startDate)) {
      startDatePicker.style.backgroundColor = 'red'; // 设置背景颜色为红色
    } else {
      startDatePicker.style.backgroundColor = ''; // 移除背景颜色样式
    }
});

//自動判斷資料夾
document.addEventListener('DOMContentLoaded', function() {
    var recordTypeSelect = document.getElementById('Record_type');
    var selectedRecordType = recordTypeSelect.value;
    sendSelectedRecordType(selectedRecordType); 

    recordTypeSelect.addEventListener('change', function() {
        var selectedRecordType = recordTypeSelect.value;
        sendSelectedRecordType(selectedRecordType);
    });
});

function sendSelectedRecordType(recordType) {
    var formData = new FormData();
    formData.append('Record_type', recordType);
    var request = new XMLHttpRequest();
    request.open('POST', '/get_folders');
    request.onload = function() {
        if (request.status === 200) {
            var folders = JSON.parse(request.responseText);
            updateSelectOptions(folders);
        } else {
            console.error('Failed to get folders');
        }
    };
    request.onerror = function() {
        console.error('Failed to get folders');
    };
    request.send(formData);
}

function updateSelectOptions(folders) {
    var selectIDSelect = document.getElementById('Select_ID');
    selectIDSelect.innerHTML = ''; 

    folders.forEach(function(folder) {
        var option = document.createElement('option');
        option.text = folder;
        option.value = folder;
        selectIDSelect.appendChild(option);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    var recordTypeSelect = document.getElementById('Record_type');
    var selectIDSelect = document.getElementById('Select_ID');
    var startDatePicker = document.getElementById('start-datePicker');

    // 獲取各個元素的值
    function getSelectedValues() {
        var recordType = recordTypeSelect.value;
        var selectID = selectIDSelect.value;
        var startDate = startDatePicker.value;

        return {
            'Record_type': recordType,
            'Select_ID': selectID,
            'start_date': startDate, // 注意這裡的屬性名與Flask中的對應
        };
    }

    // 按下按鈕時處理函式
    document.getElementById('start').addEventListener('click', function() {
        event.preventDefault(); 
        var selectedValues = getSelectedValues();
        sendSelectedValues(selectedValues);
    });
});

function sendSelectedValues(selectedValues) {
    var request = new XMLHttpRequest();
    request.open('POST', '/process_data');
    request.setRequestHeader('Content-Type', 'application/json');
    request.onload = function() {
        if (request.status === 200) {
            var responseData = JSON.parse(request.responseText);
            var newImageFilename = responseData.new_image_filename;
            var newImageURL = "static/" + newImageFilename;
            reloadImage(newImageURL);
            var imgContainer = document.querySelector('.img-container');
            var imgState = "{{ img_State }}"; 
            imgState = "display: flex;";
            imgContainer.style = imgState;
        } else {
            console.error('Failed to send data');
        }
    };
    request.onerror = function() {
        console.error('Data sending failed');
    };
    request.send(JSON.stringify(selectedValues));
}

function reloadImage(newImageURL) {
    var imgElement = document.getElementById("plotImage");
    imgElement.src = newImageURL;
}
