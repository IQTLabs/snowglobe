$(document).ready(function(){

    var globals = {};
    globals.count = -1;

    $("#connect").on("click", setup);
    $("#submit").on("click", send_answer);

    // Get API path
    function api_path(answer=false) {
	path = window.location.pathname;
	dir = path.substr(0, path.lastIndexOf("/"));
	if (!answer) {
	    return dir + "/read/" + globals.label + "/" + globals.count;
	} else {
	    return dir + "/post/" + globals.label;
	}
    };

    // Set up
    function setup() {
        globals.label = $("#id").val();
        get_prompt();
    }

    // Get prompt from server
    async function get_prompt() {
	globals.count += 1;
	while (true) {
	    const response = await fetch(api_path(false), {method: "GET"});
	    json = await response.json();
	    if (Object.keys(json).length > 0) {
		if (globals.count == 0) {
		    $("#name").val(json["name"]);
		    $("#id").attr("readonly", "readonly");
		    $("#connect").attr("disabled", "disabled");
		    $("#prompt, #answer").addClass("waiting");
		    globals.count += 1;
		} else {
		    if (json["name"] == $("#name").val()) {
			globals.count += 1;
		    } else {
			$("#prompt").val(json["content"]);
			$("#submit").removeAttr("disabled");
			$("#prompt, #answer").removeClass("waiting");
			break;
		    }
		}
	    }
	    await new Promise(r => setTimeout(r, 2000));
	}
    }

    // Send answer to server
    async function send_answer() {
	$("#submit").attr("disabled", "disabled");
	$("#prompt, #answer").addClass("waiting");
	message = {
	    "content": $("#answer").val(),
	    "format": "plaintext",
	    "name": $("#name").val(),
	    "stamp": Date(),
	    "avatar": "human.png"
	};
	await fetch(api_path(true), {
	    method: "POST",
	    headers: {"Content-Type": "application/json"},
	    body: JSON.stringify(message)
	});
	get_prompt();
    }

});
