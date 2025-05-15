import 'package:flutter/material.dart';
import 'JibJobHomePageFirstTime1.dart';
import 'Prointo1State.dart';
import 'Client1State.dart' ;

class JibJobHomePageFirstTime1State extends State<JibJobHomePageFirstTime1> {

String selectedCategory = '';

  final Color darkPurple = Color(0xFF20004E);
  final Color darkBlue = Color(0xFF003366);  // Dark blue color for selected card

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(height: 70) ,
            Container( alignment: Alignment.topCenter ,
                child : Image.asset(
                  'assets/logo.png',  // Use the logo image you want to display
                  //height: 0,
                  width: 300,
                ) ),
            SizedBox(height: 30),
            Container(alignment: Alignment.topCenter , 
            child : Text(
              "Êtes vous :",
              style: TextStyle(
                fontSize: 40,
                fontWeight: FontWeight.bold,
                color: darkPurple,
              ),
            )),
            SizedBox(height: 40),

            // Professionnel Card
            GestureDetector(
              onTap: () {
                setState(() {
                  if (selectedCategory != "Professionnel"){
                  selectedCategory = "Professionnel";
                  }else{
                    selectedCategory = "" ;
                  }
                });
              },
              child: Card(
                elevation: 4,
                margin : EdgeInsets.fromLTRB(50, 0, 50, 0) ,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: BorderSide(color: darkPurple, width: 1),
                ),
                color: selectedCategory == "Professionnel" ? Colors.deepPurpleAccent[400] : Colors.white, // Change color if selected
                child: Column(
                  children: [
                    
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        "Professionnel ?",
                        style: TextStyle(
                          fontSize: 25,
                          fontWeight: FontWeight.bold,
                          color: selectedCategory == "Professionnel" ? Colors.white : darkPurple, // Text color based on selection
                        ),
                      ),
                    ),
                    Image.asset(
                      'assets/Professionnel.png',  // Replace with actual image
                      height: 150,
                      width: double.infinity,
                    ),
                  ],
                ),
              ),
            ),

            SizedBox(height : 50 ) ,

            // Client Card
            GestureDetector(
              onTap: () {
                setState(() {
                  if (selectedCategory != "Client"){
                  selectedCategory = "Client";
                  }else{
                    selectedCategory = "" ;
                  }
                });
              },
              child: Card(
                elevation: 4,
                margin : EdgeInsets.fromLTRB(50, 0, 50, 0) ,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: BorderSide(color: darkPurple, width: 1),
                ),
                color: selectedCategory == "Client" ? Colors.deepPurpleAccent[400] : Colors.white, // Change color if selected
                child: Column(
                  children: [
                    
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        "Client",
                        style: TextStyle(
                          fontSize: 25,
                          fontWeight: FontWeight.bold,
                          color: selectedCategory == "Client" ? Colors.white : darkPurple, // Text color based on selection
                        ),
                      ),
                    ),
                    Image.asset(
                      'assets/client.png',  // Replace with actual image
                      height: 150,
                      width: double.infinity,
                    ),
                  ],
                ),
              ),
            ),

           // SizedBox(height : 80) ,

            // Show Visiteur Card only if no category is selected
            if (selectedCategory.isEmpty)
              GestureDetector(
                onTap: () {
                  
                },

                child : Container ( alignment: Alignment.bottomCenter ,
                width: 400, 
                child: Card(
                  color : Colors.white ,
                  elevation: 4,
                  margin: EdgeInsets.symmetric(vertical: 10),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: BorderSide(color: darkPurple, width: 1),
                  ),
                  
                      child : Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                          "Visiteur",
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            
                          ),
                        ),
                      ),
                )),
              ),
              
              if (selectedCategory.isNotEmpty)

              Align(
              alignment: Alignment.bottomRight,
              child: Container(
                margin : EdgeInsets.fromLTRB(0, 0, 20, 0) ,
                height: 70,
                width: 70,
                decoration: BoxDecoration(
                  color: Colors.deepPurple,
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: Icon(Icons.arrow_forward, color: Colors.white),
                  onPressed: () {
                    if (selectedCategory == "Client") {
                      runApp(
                      MaterialApp(
                        home : Client1State() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                    }else if (selectedCategory == "Professionnel"){
                      runApp(
                      MaterialApp(
                        home : Prointo1State() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                    }
                  },
                ),
              ),
            )
              ,
          ],
        ),
      ),
);
  }

}