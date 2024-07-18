var mediaRecorder;
var chunks = [];
var videoElement = document.getElementById('videoElement');
var videoDevicesSelect = document.getElementById('videoDevices');
var StartButton = document.getElementById('Startbutton');
var StopButton = document.getElementById('Stopbutton');
var statusText = document.getElementById('statusText');
var audio = null;


var data = {
  "Face": "",
  "Pose": "",
  "Pose_Count": 0,
  "Start": true,
  "SID": "",
};

const socket = io("https://10.152.196.230:8051")

socket.on('connect', function(){
  console.log("Connected...!", socket.connected)
});

socket.on('get_SID', function(data){
  document.getElementById('SID').value = data.SID;
});


//載入攝像機
window.onload = function() {
  var consent = confirm("是否引許開啟攝像機");
  if (consent) {
      getVideoDevices();
  } else {
      alert("預覽器不支援。");
  }
};
//抓取設備鏡頭裝置
async function getVideoDevices() {
  let videoDevicesSelect = document.getElementById("videoDevices")
  if (!videoDevicesSelect) {
      console.log("找不到選擇清單")
      return
  }
  let devices = await navigator.mediaDevices.enumerateDevices();
  devices.forEach((device) => {
      if (device.kind === "videoinput") {
          let option = document.createElement("option")
          option.value = device.deviceId
          option.text =
              device.label || "Camera " + (videoDevicesSelect.length + 1)
          videoDevicesSelect.appendChild(option)
          console.log("找到鏡頭:", option.text)
      }
  })
}

socket.on('play_sound', function(data) {
  const soundUrl = data.sound;
  audio = new Audio(soundUrl);
  audio.oncanplay = function() {
    if (audio.duration > 0) {
      console.log("播放");
      audio.volume = 1;
      audio.play();
    }
  };
});


socket.on('play_Pose', function(data) {
  const soundUrl = data.sound;
  audio = new Audio(soundUrl);
  audio.oncanplay = function() {
    if (audio.duration > 0) {
      audio.volume = 1;
      audio.play();
    }
  };

  // 在音檔播放完畢後執行錄影
  audio.onended = function() {
    
    console.log("音檔播放完畢，執行錄影");
    var selectedVideoDevice = videoDevicesSelect.value;
    navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: selectedVideoDevice,
      },
      audio: true,
    })
      .then(function (stream) {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = function (event) {
          chunks.push(event.data);
        };
        mediaRecorder.start();
        // 3 秒後停止錄製
        setTimeout(function() {
          stopRecording();
        }, 3000); // 3 秒

      })
      .catch(function (error) {
        console.log('無法讀取Cam:', error);
      });
  };
});


//開始辨識
async function startRecording() {

  recording = true
  console.log("影像識別 : "+recording)
  StartButton.disabled = true

  var selectedVideoDevice = videoDevicesSelect.value;
  navigator.mediaDevices.getUserMedia({
    video: {
      deviceId: selectedVideoDevice,
    }
  })
    .then(function (stream) {
      videoElement.srcObject = stream;
    })
    .catch(function (error) {
      console.log('無法讀取Cam:', error);
    });

  setInterval(() => {
    if (!recording) return;
    // 創建一個canvas元素
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');

    // 將視訊畫面繪製到canvas上
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // 將canvas上的畫面轉換為JPEG格式的圖片數據URL
    var imageData = canvas.toDataURL("image/jpeg");
    socket.emit("image", imageData)
  }, 250 )
}


//停止錄影
async function stopRecording() {
  recording = false
  console.log("停止錄影")
  StartButton.disabled = false;
  socket.emit("Reset", data);
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    audio = new Audio('/static/Finish.wav');
    audio.oncanplay = function() {
      if (audio.duration > 0) {
        audio.volume = 1;
        audio.play();
      }
    };
      mediaRecorder.stop();
      mediaRecorder.onstop = function () {
        var blob = new Blob(chunks, { type: 'video/webm' });
        var formData = new FormData();
        var SID = document.getElementById('SID').value;
        var Record_type = document.getElementById('Record_type').value;
        var request = new XMLHttpRequest();
        chunks = [];
        formData.append('video', blob, 'recorded_video.webm');
        formData.append('SID', SID);
        formData.append('Record_type', Record_type);
        request.open('POST', '/upload_video'); 
        request.onload = function() {
          console.log('影片上傳成功');
        };
        request.onerror = function() {
          console.error('影片上傳失敗');
        };
        request.send(formData);
      };
  }
}
