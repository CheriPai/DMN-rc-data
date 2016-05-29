var correct = 0;
var total = 0;

$("#randTest, #randValidation").click(function() {
    if (this.id === "randTest") {
        var dataset = "test";
    }
    else {
        var dataset = "validation";
    }
    $.getJSON("/random/" + dataset, {
    }, function(data) {
        ++total;
        $("#context").text(data["context"]);
        $("#correct").text(data["correct_answer"]);
        $("#prediction").text(data["prediction"]);
        $("#question").text(data["question"]);

        if (data["correct_answer"] === data["prediction"]) {
            $("#prediction").css("border-color", "rgba(0, 204, 0, 1)");
            ++correct;
        }
        else {
            $("#prediction").css("border-color", "red");
        }
        $("#accuracy").text(Math.round(correct/total*10000)/100 + "%");
    });
});
