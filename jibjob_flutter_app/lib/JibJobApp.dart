import 'package:flutter/material.dart';
import 'screens/SplashScreen.dart';
import 'screens/Profileclient.dart' ;
import 'screens/JibJobAuthPage.dart' ;
import 'screens/Profilepro.dart' ;
import 'screens/listeServices.dart' ;
import 'screens/ProSignUpPage0.dart' ;

class JibJobApp extends StatelessWidget{
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home : SplashScreen() ,
      debugShowCheckedModeBanner: false
    ) ;
    
  }
  
  listeServices() {}
}