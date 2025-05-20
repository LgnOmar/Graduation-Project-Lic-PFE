import 'package:flutter/material.dart';
import 'package:auto_size_text/auto_size_text.dart';
import 'Client1State.dart' ;
import 'Client3State.dart' ;
import 'JibJobHomePageFirstTime1.dart';




class Client2State extends StatelessWidget {


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

            Container(
                  width: screenWidth ,
                  height: screenHeight * 0.3,
                  child: Image.asset(
                    "assets/client2.png",
                    fit: BoxFit.contain,
                  ),
                ),

                // Text
                Container(
  height: screenHeight * 0.19, // Container height
  alignment: Alignment.topCenter,
  width: screenWidth * 0.7, // Container width
  child: AutoSizeText(
    'Les professionnels vous répondent',
    style: TextStyle(
      fontWeight: FontWeight.w600,
      color: Colors.indigo[900],
      fontSize: screenHeight * 0.06 ,
    ),
    maxLines: 2,  // Number of lines of text
    
  ),
),


                    SizedBox(height: screenHeight* 0.03) ,

                    Container(
                        height : screenHeight * 0.09 ,
                        width: screenWidth * 0.7,
                        child : AutoSizeText(
                            "Comparez les offres venant des professionnels de votre commune" ,
                             style: TextStyle(
                              fontSize: screenHeight * 0.022, 
                              fontWeight: FontWeight.w400 , 
                              color: Colors.indigo[900]
                        ) ,
                        maxLines: 3,
                        ))

                  


         

        ]
      ),

       Positioned(
                bottom: screenHeight * 0.01 , right: screenWidth * 0.05 ,
                child :
                Container(
                  height : screenHeight * 0.09 ,
                  alignment: Alignment.bottomRight ,
                  child:

                  IconButton(
                    icon:
                    Image.asset(
                      'assets/forward.png',
                      fit: BoxFit.fitHeight,
                    ),

                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Client3State()),
                      );
                    },
                  ),
                )
            ),

            Positioned(
                bottom: screenHeight * 0.01 , left: screenWidth * 0.05 ,
                child :
                Container(
                  height : screenHeight * 0.09 ,
                  alignment: Alignment.bottomRight ,
                  child:

                  IconButton(
                    icon:
                    Image.asset(
                      'assets/backward.png',
                      fit:BoxFit.fitHeight ,
                    ),

                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Client1State()),
                      );
                    },
                  ),
                )
            ),

      ]
      
      
      ));
      }}