$("#randTest, #randValidation").click(function() {
    if (this.id === "randTest") {
        var dataset = "test";
    }
    else {
        var dataset = "validation";
    }
    $.getJSON("/random/" + dataset, {
    }, function(data) {
        $("#context").text(data["context"]);
        $("#correct").text(data["correct_answer"]);
        $("#prediction").text(data["prediction"]);
        $("#question").text(data["question"]);

        if (data["correct_answer"] === data["prediction"]) {
            $("#prediction").css("border-color", "rgba(0, 204, 0, 1)");
        }
        else {
            $("#prediction").css("border-color", "red");
        }
    });
});
