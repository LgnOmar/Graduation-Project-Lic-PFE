import 'package:flutter/material.dart';
import 'Client2State.dart' ;
import 'ClientSignUpPage.dart' ;

class Client3State extends StatelessWidget {
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
            SizedBox(height: 70),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset(
                  'assets/logo.png',
                  width: 300,
                ),
              ],
            ),
            SizedBox(height: 70),

            Container( alignment: Alignment.topCenter, 
            child : Image.asset(
              'assets/client3.png',
              height: 250,
              width: double.infinity,
              fit: BoxFit.contain,
            )) ,
            SizedBox(height: 40),

            Container(margin :EdgeInsets.fromLTRB(30, 0, 0, 0)   ,
            child : Text(
              "Choisissez    la meilleure offre",
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
              "Sélectionnez celui que vous souhaitez rencontrer",
              textAlign: TextAlign.center,
              style: TextStyle(
                color: const Color(0xFF130160) /* Main1 */,
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
                        home : Client2State() ,
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
                      home :ClientSignUpPage() ,
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