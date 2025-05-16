import 'package:flutter/material.dart';
import 'Prointo2State.dart' ;
import 'ProSignUpPage0.dart' ;

class Prointo3State extends StatelessWidget {
  final Color darkPurple = Color(0xFF20004E);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            //SizedBox(height: 70),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset(
                  'assets/logo.png',
                  width: 300,
                ),
              ],
            ),
            //SizedBox(height: 70),

            Container( alignment: Alignment.topCenter, 
            child : Image.asset(
              'assets/pro3.png', // Replace with actual image
              height: 250,
              width: double.infinity,
              fit: BoxFit.contain,
            )) ,
            SizedBox(height: 40),

            Container( alignment: Alignment.center , margin :EdgeInsets.fromLTRB(70, 0, 30, 0)   ,
            child : Text(
              "Rejoindre la communauté",
              style: TextStyle(
                color: const Color(0xFF130160),
                fontSize: 45,
                fontFamily: 'DM Sans',
                fontWeight: FontWeight.w600,
                height: 0.95,
              ),
            )),
            SizedBox(height: 35),
            Container( margin :EdgeInsets.fromLTRB(40, 0, 40, 0) , 
            child:Text(
              "Collecter des feedbacks, échanger avec votre marché et partager des connaissances avec des collègues Créer un compte professionnel",
              textAlign: TextAlign.center,
              style: TextStyle(
                color: const Color(0xFF130160) ,
                fontSize: 22,
                fontFamily: 'DM Sans',
                fontWeight: FontWeight.w300,
                height: 0.95,
              ),
            )),
            SizedBox(height: 60),
            
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                
                Align(
              alignment: Alignment.bottomRight,
              child: Container(
                margin : EdgeInsets.fromLTRB(0, 0, 0, 0) ,
                height: 70,
                width: 70,
                decoration: BoxDecoration(
                  color: Colors.deepPurple,
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: Icon(Icons.arrow_back, color: Colors.white),
                  onPressed: () {
                      runApp(
                      MaterialApp(
                        home : Prointo2State() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                  },
                ),
              ),
            ),


                Align(
              alignment: Alignment.bottomRight,
              child: Container(
                margin : EdgeInsets.fromLTRB(0, 0, 0, 0) ,
                height: 70,
                width: 70,
                decoration: BoxDecoration(
                  color: Colors.deepPurple,
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: Icon(Icons.arrow_forward, color: Colors.white),
                  onPressed: () {
                      runApp(
                      MaterialApp(
                        home :ProSignUpPage0() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                  },
                ),
              ),
            ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}