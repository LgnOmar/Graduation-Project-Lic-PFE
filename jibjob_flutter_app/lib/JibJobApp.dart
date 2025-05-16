import 'package:flutter/material.dart';
import 'SplashScreen.dart';
import 'Profileclient.dart' ;
import 'JibJobAuthPage.dart' ;
import 'Profilepro.dart' ;
import 'listeServices.dart' ;
import 'ProSignUpPage0.dart' ;

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