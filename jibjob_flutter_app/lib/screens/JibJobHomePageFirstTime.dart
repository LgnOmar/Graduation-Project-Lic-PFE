import 'package:flutter/material.dart';
import 'JibJobHomePageFirstTime1.dart' ;

class JibJobHomePageFirstTime extends StatelessWidget {
  final Color darkPurple = Color(0xFF20004E);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(0),
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
            SizedBox(height: 40),

            Container( alignment: Alignment.topCenter, 
            child : Image.asset(
              'assets/illustration.png', // Replace with actual image
              height: 300,
              width: double.infinity,
              fit: BoxFit.contain,
            )) ,
            //SizedBox(height: 80),

            Container( alignment: Alignment.centerLeft , margin :EdgeInsets.fromLTRB(30, 0, 0, 0) ,
            child : Text(
              "Trouvez Votre",
              style: TextStyle(
                fontSize: 50,
                fontFamily: 'DM Sans',
                fontWeight: FontWeight.w600,
                height: 0.95,
              ),
            )),

            SizedBox(height : 5) ,

           Container( alignment: Alignment.centerLeft , margin :EdgeInsets.fromLTRB(30, 0, 0, 0) ,
            child : Row(
            children : [Text(
              "Dream Job",
              style: TextStyle(
                color: const Color.fromRGBO(109, 66, 244, 1),
                decoration: TextDecoration.underline,
                fontSize: 50,
                fontFamily: 'DM Sans',
                fontWeight: FontWeight.w700,
                height: 0.95,
              ),
            ),
            
            Text(
              " Ici!",
              style: TextStyle(
                fontSize: 40,
                fontFamily: 'DM Sans',
                fontWeight: FontWeight.w600,
                height: 0.95,
              ),
            )
            
            ])),

            SizedBox(height: 60),

            Container( margin: EdgeInsets.fromLTRB(20, 0, 10, 0)  ,
            child:Text(
              "Explorez tous les postes les plus passionnants en fonction de vos intérêts et de votre spécialisation.",
              style: TextStyle(
                color: const Color.fromRGBO(82, 75, 107, 1),
                fontSize: 16,
                fontFamily: 'DM Sans',
                fontWeight: FontWeight.w400,
                height: 0.95,
              ),
            )),

            SizedBox(height: 60),


                Align(
              alignment: Alignment.bottomRight,
              child: Container(
                margin : EdgeInsets.fromLTRB(0, 0, 40, 0) ,
                height: 70,
                width: 70,
                decoration: BoxDecoration(
                  color: const Color.fromRGBO(19, 1, 96, 1),
                  shape: BoxShape.circle,
                ),
                child: IconButton(
                  icon: Icon(Icons.arrow_forward, color: Colors.white),
                  iconSize: 25,
                  onPressed: () {
                      runApp(
                      MaterialApp(
                        home :JibJobHomePageFirstTime1() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                  },
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}