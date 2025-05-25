import 'package:flutter/material.dart';


class SignIn extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final TextEditingController emailController = TextEditingController();
    final TextEditingController passwordController = TextEditingController();

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: Icon(Icons.arrow_back, color: Colors.deepPurple),
        title: Text("Compte", style: TextStyle(color: Colors.black)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24.0),
        child: ListView(
          children: [
            SizedBox(height: 20),
            Text("Email", style: TextStyle(fontWeight: FontWeight.bold)),
            SizedBox(height: 8),
            TextField(
              controller: emailController,
              decoration: InputDecoration(
                hintText: "Ex: zakarya@gmail.com",
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 16),
              ),
            ),
            SizedBox(height: 16),
            Text("Mot de pass", style: TextStyle(fontWeight: FontWeight.bold)),
            SizedBox(height: 8),
            TextField(
              controller: passwordController,
              obscureText: true,
              decoration: InputDecoration(
                hintText: "********",
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 16),
              ),
            ),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () {},
                child: Text("Mot de pass oublié ?", style: TextStyle(color: Colors.purple)),
              ),
            ),
            SizedBox(height: 16),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo[900],
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              onPressed: () {},
              child: Text("Se connecter", style: TextStyle(fontSize: 16)),
            ),
            SizedBox(height: 16),
            Center(child: Text("Vous n'avez pas encore de compte ?")),
            SizedBox(height: 8),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepPurpleAccent,
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
              onPressed: () {},
              child: Text("Creer un compte", style: TextStyle(fontSize: 16)),
            ),
            SizedBox(height: 16),
            Row(children: <Widget>[
              Expanded(child: Divider()),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8.0),
                child: Text("Ou"),
              ),
              Expanded(child: Divider()),
            ]),
            SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: () {},
              icon: Image.asset('assets/google.png', height: 24), // Add asset image
              label: Text("Continuez avec google"),
              style: OutlinedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () {},
              icon: Image.asset('assets/facebook.png', height: 24), // Add asset image
              label: Text("Continuez avec Facebook"),
              style: OutlinedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}
