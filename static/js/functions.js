function preview_image(event) 
{
	var reader = new FileReader();
	reader.onload = function()
 {
	var output = document.getElementById('preview');
	output.src = reader.result;
 }
	reader.readAsDataURL(event.target.files[0]);
}

function clearall()
{
	document.getElementById('preview').src = "";
	document.getElementById('detection').src ="";
}