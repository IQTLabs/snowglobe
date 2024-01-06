$(document).ready(function(){

    var globals = {};
    globals.count = -1;

    $("#connect").on("click", setup);
    $("#submit").on("click", send_answer);

    // Get path to API
    function api_path(answer=false) {
	return '/' + (answer ? 'answer' : 'prompt')
	    + '/' + globals.label + '/' + globals.count;
    };

    // Set up
    function setup() {
        globals.label = $("#id").val();
        get_prompt();
    }

    // Get prompt from server
    async function get_prompt() {
	globals.count += 1;
	const response = await fetch(api_path(false), {
	    method: "GET"
	});
	json = await response.json()
	console.log(json)
	while (true) {
	    break;
	}
    }

    // Send answer to server
    async function send_answer() {
    }

});
