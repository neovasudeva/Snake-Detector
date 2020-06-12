// event handler for submit button
async function submit() {
  // get uploaded image and output container
  let imgs = document.getElementById("file-upload").files;

  // check if image was uploaded at all
  if (imgs.length == 0) {
    // tell user an image needs to be uploaded
    alert("Please upload an image before submitting.");
    return;
  }

  // send data to server 
  let img = imgs[0];
  let formData = new FormData();
  formData.append(img.name, img);

  // get response from server
  const url = await fetch("/home", {
    method: "POST",
    body: formData
  }).then(function(response) {
    console.log(response);
    return response.blob();
  }).then(function(blob) {
    return URL.createObjectURL(blob);
  });

  // create child element
  let mod_img = document.getElementById('model-img');
  while (mod_img.firstChild) {
    mod_img.removeChild(mod_img.firstChild);
  }
  let new_img = document.createElement("IMG");
  new_img.setAttribute("alt", "model_img");
  new_img.src = await url;
  mod_img.appendChild(new_img);
}

// event handler for upload file
function upload() {
  // get uploaded file's filename and label
  let label = document.getElementsByClassName("custom-file-label")[0];
  let file = document.getElementById("file-upload").files[0];

  // verify JPG extension
  if (file.type != "image/jpeg") {
    console.log(file.type);
    alert("Please upload JPG files only.");
    return;
  }

  // change label
  label.innerHTML = file.name;

  // preview the image
  let disp_img = document.getElementById("display-img");

  // wipe all child nodes from display container
  while (disp_img.firstChild) {
    disp_img.removeChild(disp_img.firstChild);
  }

  // display uploaded image
  let new_img = document.createElement("IMG");
  new_img.setAttribute("alt", "#");
  new_img.setAttribute("src", URL.createObjectURL(file));
  disp_img.appendChild(new_img);
}

// event listeners
document.getElementById("upload-btn").addEventListener("click", submit);
document.getElementById("file-upload").addEventListener("change", upload);


