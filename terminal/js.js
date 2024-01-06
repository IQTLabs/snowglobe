$(document).ready(function(){

    var globals = {};
    globals.count = -1;
    globals.label = 9999;

    $("#connect").on("click", setup);
    $("#submit").on("click", send_answer);

    // Get path to API
    function get_path(answer=false) {
	
    };

    // Set up
    function setup() {
        console.log("in setup");
        globals.label = $("#id").val();
        get_prompt();
    }

    // Get prompt from server
    function get_prompt() {
	globals.count += 1;
	while (true) {
	    console.log(globals.label);
	    break;
	}
    }

    // Send answer to server
    function send_answer() {
    }

});
