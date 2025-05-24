import 'package:flutter/material.dart';
import '../JibJobApp.dart';
import 'JibJobHomePageFirstTime.dart';



class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    navigateToMainPage();
  }

  // Function to delay for 3 seconds and then navigate to the main page
  void navigateToMainPage() {
    Future.delayed(Duration(seconds: 3), () {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) {
            return JibJobHomePageFirstTime();
          },
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            // Slide transition
            const begin = Offset(0.0, -1.0); // Starting position: off-screen above
            const end = Offset.zero; // Ending position: screen center
            const curve = Curves.easeInOut;

            var tween = Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
            var offsetAnimation = animation.drive(tween);

            return SlideTransition(position: offsetAnimation, child: child);
          },
        ),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xD3200272),
      body: Center(
        child: Container(alignment: Alignment.center ,  width: 700, height: 700 , child: Image.asset('assets/logo2.png' , fit: BoxFit.cover,)),
      ),
    );
  }
}