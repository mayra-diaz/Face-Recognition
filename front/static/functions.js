$("#loading").hide();

function loading() {
  $("#loading").show();
}

$(function () {
  $("#flash")
    .delay(500)
    .fadeIn("normal", function () {
      $(this).delay(5000).fadeOut();
    });
});

var rad = document.myForm.type;
rad[0].addEventListener("change", function () {
  $("#numElementLabel").show();
  $("#numElement").show();
  $("#numRangeLabel").hide();
  $("#numRange").hide();
});

rad[1].addEventListener("change", function () {
  $("#numElementLabel").hide();
  $("#numElement").hide();
  $("#numRangeLabel").show();
  $("#numRange").show();
});
