import 'package:flutter/material.dart';
import 'package:jibjob/Person.dart';
import 'package:jibjob/Pro.dart';
import 'package:jibjob/screens/Client/MakeOrder.dart';
import 'package:jibjob/screens/JibJobHomePageFirstTime1.dart';
import 'package:jibjob/screens/Pro/Profilepro.dart';
class SignInPage extends StatelessWidget {

  final emailController = TextEditingController();
  final passwordController = TextEditingController();

   @override
  void dispose() {
    emailController.dispose();
    passwordController.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Color darkPurple = Color(0xFF20004E);
    final Color lightPurple = Color(0xFF7B61FF);

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.deepPurple),
          onPressed: () => Navigator.of(context).pop(),
        ),
        centerTitle: true,
        title: Text(
          "Compte",
          style: TextStyle(
            color: Colors.deepPurple,
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.symmetric(horizontal: 24, vertical: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(height: 16),
            Text(
              "Email",
              style: TextStyle(fontWeight: FontWeight.w500, fontSize: 16),
            ),
            SizedBox(height: 4),
            TextField(
              controller: emailController ,
              decoration: InputDecoration(
                hintText: "Ex: zakarya@gmail.com",
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            SizedBox(height: 16),
            Text(
              "Mot de pass",
              style: TextStyle(fontWeight: FontWeight.w500, fontSize: 16),
            ),
            SizedBox(height: 4),
            TextField(
              controller: passwordController ,
              obscureText: true,
              decoration: InputDecoration(
                hintText: "********",
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
              ),
            ),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () {},
                child: Text(
                  "Mot de pass oublié ?",
                  style: TextStyle(
                    color: lightPurple,
                    fontSize: 13,
                  ),
                ),
              ),
            ),
            SizedBox(height: 8),
            ElevatedButton(
              onPressed: () {
                if (Liste_Pros.isNotEmpty){
                  for (int i = 0 ; i < Liste_Pros.length ; i++){ {
                    if (Liste_Pros[i].email == emailController.text && Liste_Pros[i].password == passwordController.text) {
                      
                            print("nb clients${Person.nb}\n");
                            print("nb Pros${Pro.nb}\n");

                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Profilepro(Pro_Position : i)),
                      );
                      return;
                    }
                  }
                }
              }

              if (Liste_Clients.isNotEmpty){
                  for (int i = 0 ; i < Liste_Clients.length ; i++){ {
                    if (Liste_Clients[i].email == emailController.text && Liste_Clients[i].password == passwordController.text) {
                      
                            print("nb clients${Person.nb}\n");
                            print("nb Pros${Pro.nb}\n");
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => (MakeOrder(Client_Position : i))),
                      );
                      return;
                    }
                  }
                }
              }

            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Email ou mot de passe incorrect !')),
            );



              
              
              }
              ,
              style: ElevatedButton.styleFrom(
                backgroundColor: darkPurple,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                padding: EdgeInsets.symmetric(vertical: 14),
              ),
              child: Text(
                "Se connecter",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            SizedBox(height: 12),
            Center(
              child: Text(
                "Vous n'avez pas encore de compte ?",
                style: TextStyle(fontSize: 13, color: Colors.black87),
              ),
            ),
            SizedBox(height: 8),
            ElevatedButton(
              onPressed: () {
                print("Email :${emailController.text}\n");
                print("Password :${passwordController.text}\n");
                Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => JibJobHomePageFirstTime1()),
                      );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: lightPurple,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                padding: EdgeInsets.symmetric(vertical: 14),
              ),
              child: Text(
                "Creer un compte",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Expanded(child: Divider(thickness: 1)),
                Padding(
                  padding: EdgeInsets.symmetric(horizontal: 8),
                  child: Text("Ou", style: TextStyle(color: Colors.black54)),
                ),
                Expanded(child: Divider(thickness: 1)),
              ],
            ),
            SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: () {},
              icon: Image.asset(
                'assets/google.png',
                height: 24,
                width: 24,
              ),
              label: Text(
                "Continuez avec google",
                style: TextStyle(color: Colors.black87, fontWeight: FontWeight.w500),
              ),
              style: OutlinedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 12),
                side: BorderSide(color: Colors.grey.shade300),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                backgroundColor: Colors.white,
              ),
            ),
            SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () {},
              icon: Image.asset(
                'assets/facebook.png',
                height: 24,
                width: 24,
              ),
              label: Text(
                "Continuez avec Facebook",
                style: TextStyle(color: Colors.black87, fontWeight: FontWeight.w500),
              ),
              style: OutlinedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 12),
                side: BorderSide(color: Colors.grey.shade300),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                backgroundColor: Colors.white,
              ),
            ),
            SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}