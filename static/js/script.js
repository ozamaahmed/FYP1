$(document).ready(function() { //html document ready for work hue tak rukta
    $('#searchForm').submit(function(event) { // searchForm naam ke form ku select kiya gaya hai  using this 
        event.preventDefault(); // default harkat hone se roko
        
        // Get the form data
        var formData = $(this).serialize(); // data ko serialize karo
        
        // Send the AJAX request
        $.ajax({ // ajax ka istemal hua jaa raha hai aur /process call kia ja raha hai post method se 
            url: '/process',
            type: 'POST', 
            data: formData, // data is formdata
            success: function(response) {  // 
                // Clear previous results
                $('#output').empty();
                
                // Display the results
                if (response.output.length > 0) { 
                    var resultsHtml = ''; 
                    
                    response.output.forEach(function(result) { 
                        var timestamp = result.split(' -> ')[0]; // this js code also seperates the string with the -> and then puts timestamp and subtitle in different classes
                        var subtitle = result.split(' -> ')[1];  
                        
                        resultsHtml += '<p><span class="timestamp">' + timestamp + '</span> -> ' + subtitle + '</p>'; // this is some html code that we are making dynamically 
                    });
                    console.log(resultsHtml)
                    
                    $('#output').html(resultsHtml); 
                } else {
                    $('#output').text('No matching subtitles found.');
                }
            },
            error: function() {
                $('#output').text('An error occurred. Please try again.');
            }
        });
    });
});zz



