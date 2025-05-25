import 'package:flutter/material.dart';
import 'JibJobHomePageFirstTime1.dart' ;

class JibJobHomePageFirstTime extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // Get the screen width and height
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
                SizedBox(height: screenHeight* 0.05) ,

                Container(
                  width: screenWidth ,
                  height: screenHeight * 0.4,
                  child: Image.asset(
                    "assets/illustration.png",
                    fit: BoxFit.contain,
                  ),
                ),
                SizedBox(height: screenHeight* 0.03) ,

                // Text
                Align(
                    alignment: Alignment.topLeft ,
                    child :Container(
                        padding:EdgeInsets.fromLTRB(screenWidth * 0.05, 0, 0, 0),
                        //color: Colors.amber,
                        width: screenWidth * 0.97,
                        height: screenHeight * 0.33,
                        child : Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text("Trouver Votre" , style: TextStyle(fontSize: screenHeight * 0.04 , fontWeight: FontWeight.w600 ),) ,
                            Text("Dream Job Ici !" , style: TextStyle(color: Colors.deepPurpleAccent, fontSize: screenHeight * 0.04 , fontWeight: FontWeight.w600 ),) ,
                            Text("Explorez tous les postes les plus passionnants en fonction de vos intérêts et de votre spécialisation." , style: TextStyle(fontSize: screenHeight * 0.02 , fontWeight: FontWeight.w300 , fontFamily: 'DM Sans', ),) ,


                          ],
                        )
                    )),
              ],
            ),

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
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => JibJobHomePageFirstTime1()),
                      );
                    },
                  ),
                )
            ),

          ],


        ));
  }
}








