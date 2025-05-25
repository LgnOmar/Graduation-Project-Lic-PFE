import 'package:flutter/material.dart';
import 'Pro/Prointo1State.dart';
import 'Client/Client1State.dart' ;


class JibJobHomePageFirstTime1 extends StatefulWidget {
  @override
  _JibJobHomePageFirstTime1State createState() => _JibJobHomePageFirstTime1State();
}

class _JibJobHomePageFirstTime1State extends State<JibJobHomePageFirstTime1> {

  String selectedCategory = '';

  final Color darkPurple = Color(0xFF20004E);
  final Color darkBlue = Color(0xFF003366);  

  @override
  Widget build(BuildContext context) {

      double screenWidth = MediaQuery.of(context).size.width;
      double screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(

      body: Stack(
        children: [
         Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisAlignment: MainAxisAlignment.start ,
          children: [
                SizedBox(height: screenHeight* 0.08) ,
                Container(
                  alignment: Alignment.center ,
                  width: screenWidth * 0.7,
                  height: screenHeight * 0.1,
                  child: Image.asset(
                    "assets/logo.png",
                    fit: BoxFit.contain,
                  ),
                ),

            SizedBox(height: screenHeight* 0.02) ,
            Container(alignment: Alignment.topCenter ,
                child : Text(
                  "Êtes vous :",
                  style: TextStyle(
                    fontSize: screenHeight * 0.04 ,
                     fontWeight: FontWeight.w800 ,
                    color: darkPurple,
                  ),
                )),
            SizedBox(height: screenHeight* 0.03) ,

            // Professionnel Card
            Container(
            height: screenHeight * 0.19,
            child : GestureDetector(
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
                margin : selectedCategory == "Professionnel" ? EdgeInsets.fromLTRB(screenWidth*0.1, 0, screenWidth*0.1, 0) : EdgeInsets.fromLTRB(screenWidth*0.15, 0, screenWidth*0.15, 0)  ,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: BorderSide(color: darkPurple, width: 1),
                ),
                color: selectedCategory == "Professionnel" ? Colors.deepPurpleAccent[400] : Colors.white, // Change color if selected
                child: Column(
                  children: [

                      Container(
                        height : screenHeight * 0.035 ,
                      child: Text(
                        "Professionnel ?",
                        style: TextStyle(
                          fontSize: screenHeight * 0.028,
                          fontWeight: FontWeight.bold,
                          color: selectedCategory == "Professionnel" ? Colors.white : darkPurple, // Text color based on selection
                        ),
                      )),
                    Container(
                      width: double.infinity,
                      height: screenHeight * 0.15,
                    child : Image.asset(
                      'assets/Professionnel.png',
                        fit : BoxFit.fitHeight ,
                    )),
                  ],
                ),
              ),
            )),

            SizedBox(height : screenHeight * 0.04) ,

            // Client Card
            Container(
            height: screenHeight * 0.19,
            child : GestureDetector(
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
                margin : selectedCategory == "Client" ? EdgeInsets.fromLTRB(screenWidth*0.1, 0, screenWidth*0.1, 0) : EdgeInsets.fromLTRB(screenWidth*0.15, 0, screenWidth*0.15, 0)  ,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                  side: BorderSide(color: darkPurple, width: 1),
                ),
                color: selectedCategory == "Client" ? Colors.deepPurpleAccent[400] : Colors.white, // Change color if selected
                child: Column(
                  children: [

                    Container(
                      height: screenHeight * 0.035,
                      child: Text(
                        "Client",
                        style: TextStyle(
                          fontSize: screenHeight * 0.028,
                          fontWeight: FontWeight.bold,
                          color: selectedCategory == "Client" ? Colors.white : darkPurple, // Text color based on selection
                        ),
                      )),
                    Container(
                      height: 0.15 * screenHeight ,
                      width: double.infinity,
                    child : Image.asset(
                      'assets/client.png',
                      width: double.infinity,
                      fit : BoxFit.fitHeight ,
                    )),
                  ],
                ),
              ),
            )),

            SizedBox(height : screenHeight * 0.04) ,

            // Show Visiteur Card only if no category is selected
            if (selectedCategory.isEmpty)
              GestureDetector(
                onTap: () {

                },

                child : Container (
                    margin : EdgeInsets.fromLTRB(screenWidth * 0, 0, screenWidth * 0 , 0),
                    alignment: Alignment.topCenter,
                    child: Card(
                      color : Colors.white ,
                      elevation: 4,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                        side: BorderSide(color: darkPurple, width: 1),
                      ),

                      child : Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Text(
                          "         Visiteur         ",
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,

                          ),
                        ),
                      ),
                    )),
              ),
          ],
        ),
        if (selectedCategory.isNotEmpty)
        Positioned(
                bottom: screenHeight * 0.005 , right: screenWidth * 0.05 ,
                child :
                Container(
                  padding: EdgeInsets.all(screenWidth * 0.04),
                  alignment: Alignment.bottomRight ,
                  child:

                  IconButton(
                    icon:
                    Image.asset(
                      'assets/forward.png',
                    ),

                    
                    onPressed: () {
                      if (selectedCategory == 'Professionnel'){
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Prointo1State()),
                      );
                      }else if (selectedCategory == 'Client'){
                        Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Client1State()),
                      );
                      }
                    },
                  ),
                )
            ),
        ]
      ),
    );
  }

}