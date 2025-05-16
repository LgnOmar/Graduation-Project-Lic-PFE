import 'package:flutter/material.dart';
import 'JibJobAuthPage.dart';
import 'ClientSignUpPage.dart';
import 'ProSignUpPage.dart';

class CreateAccountPage extends StatelessWidget {

void onClientPressed() {
    runApp(
      MaterialApp(
      home : ClientSignUpPage() ,
      debugShowCheckedModeBanner: false
    ));
    
  }

  void onProPressed() {
    runApp(
      MaterialApp(
      //home : ProSignUpPage() ,
      debugShowCheckedModeBanner: false
    ));
  }

  final Color darkPurple = Color(0xFF20004E);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      bottomNavigationBar: BottomNavigationBar(
        selectedItemColor: darkPurple,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Accueil'),
          BottomNavigationBarItem(icon: Icon(Icons.business), label: 'Pros'),
          BottomNavigationBarItem(icon: Icon(Icons.add_circle, size: 40), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.notifications), label: 'Demandes'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Compte'),
        ],
        currentIndex: 4,
        type: BottomNavigationBarType.fixed,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header row
              Row(
                children: [
                  IconButton(
                  icon: Icon(Icons.arrow_back, color: darkPurple) ,
                  onPressed: () {
                      runApp(
                      MaterialApp(
                        home : JibJobAuthPage() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                  },
                ) ,
                  
                  SizedBox(width: 16),
                  Text(
                    'Creer un compte',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: darkPurple,
                    ),
                  ),
                ],
              ),
              SizedBox(height: 32),

              // Client card

              GestureDetector(
                onTap: onClientPressed ,
                child :
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade400),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Client",
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold)),
                    SizedBox(height: 4),
                    Text("Vous etes ici pour realiser vos projets ✅"),
                  ],
                ),
              )),



              SizedBox(height: 16),
              Center(child: Text("Ou")),
              SizedBox(height: 16),


              GestureDetector(
                onTap: onProPressed ,
                child :
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey.shade400),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("Professionnel",
                        style: TextStyle(
                            fontSize: 18, fontWeight: FontWeight.bold)),
                    SizedBox(height: 4),
                    Text("Vous etes ici pour obtenir des clients 🏆"),
                  ],
                ),
              )),
            ],
          ),
        ),
      ),
    );
  }
}