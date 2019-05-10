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
	document.getElementById('detection').src = "";
	var el1 = document.getElementById('detid');
	el1.parentNode.removeChild(el1);
	var el2 = document.getElementById('butid');
	el2.parentNode.removeChild(el2);
}
