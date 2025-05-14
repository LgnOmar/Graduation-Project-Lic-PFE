import 'package:flutter/material.dart';
import 'CreateAccountPage.dart';
import 'JibJobHomePageFirstTime.dart';

class JibJobAuthPage extends StatelessWidget {
  final Color darkPurple = Color(0xFF20004E);
  final Color lightPurple = Color(0xFF9F58FF);

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
        child: SingleChildScrollView(
          padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Back arrow + title
              Row(
                children: [

                  IconButton(
                  icon: Icon(Icons.arrow_back, color: darkPurple),
                  onPressed: () {
                    runApp(
                      MaterialApp(
                        home : JibJobHomePageFirstTime() ,
                        debugShowCheckedModeBanner: false
                      )) ;
                    
                  }) ,

                  SizedBox(width: 16),
                  Text(
                    'Compte',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: darkPurple,
                    ),
                  ),
                ],
              ),
              SizedBox(height: 32),

              // Email
              Text("Email"),
              SizedBox(height: 6),
              TextField(
                decoration: InputDecoration(
                  hintText: 'Ex: zakarya@gmail.com',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  contentPadding: EdgeInsets.symmetric(horizontal: 12),
                ),
              ),
              SizedBox(height: 16),

              // Password
              Text("Mot de pass"),
              SizedBox(height: 6),
              TextField(
                obscureText: true,
                decoration: InputDecoration(
                  hintText: '********',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  contentPadding: EdgeInsets.symmetric(horizontal: 12),
                ),
              ),

              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () {},
                  child: Text(
                    'Mot de pass oublié ?',
                    style: TextStyle(color: lightPurple),
                  ),
                ),
              ),

              // Login button
              SizedBox(height: 12),
              ElevatedButton(
                onPressed: () {},
                style: ElevatedButton.styleFrom(
                  backgroundColor: darkPurple,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  minimumSize: Size(double.infinity, 50),
                ),
                child: Text("Se connecter"),
              ),

              SizedBox(height: 16),
              Center(child: Text("Vous n’avez pas encore de compte ?")),

              // Create account
              SizedBox(height: 12),
              ElevatedButton(
                onPressed: () {

                  runApp(
                      MaterialApp(
                        home : CreateAccountPage() ,
                        debugShowCheckedModeBanner: false
                      )) ;

                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: lightPurple,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  minimumSize: Size(double.infinity, 50),
                ),
                child: Text("Creer un compte"),
              ),

              SizedBox(height: 24),
              Center(child: Text("Ou")),
              SizedBox(height: 16),

              // Google login
              OutlinedButton.icon( 
                onPressed: () {},
                icon: Icon(Icons.g_mobiledata, color: Colors.red , size : 30),
                label: Text("Continuez avec google"),
                style: OutlinedButton.styleFrom(
                  minimumSize: Size(double.infinity, 50),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
              ),

              SizedBox(height: 12),

              // Facebook login
              OutlinedButton.icon(
                onPressed: () {},
                icon: Icon(Icons.facebook, color: Colors.blue , size : 30),
                label: Text("Continuez avec Facebook"),
                style: OutlinedButton.styleFrom(
                  minimumSize: Size(double.infinity, 50),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}