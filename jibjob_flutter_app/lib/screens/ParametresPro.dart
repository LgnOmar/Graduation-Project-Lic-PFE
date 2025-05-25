import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:jibjob/screens/SignInPage.dart';

class ParametresPro extends StatefulWidget {
  @override
  State<ParametresPro> createState() => _ParametresProState();
}

class _ParametresProState extends State<ParametresPro> {
  int selectedNotif = 1; 

  @override
  Widget build(BuildContext context) {
    final Color darkPurple = Color(0xFF20004E);
    final Color green = Color(0xFF27AE60);

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: darkPurple),
          onPressed: () => Navigator.of(context).pop(),
        ),
        centerTitle: true,
        title: Text(
          "Parametres",
          style: TextStyle(
            color: darkPurple,
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Partager mon profile pro
            Text(
              "Partager mon profile pro",
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            SizedBox(height: 4),
            Text(
              "Vous êtes fière de votre profile bricoram ? Alors, vous pouvez le mentionner sur votre CV, votre carte de visite ou sur les réseaux sociaux.",
              style: TextStyle(fontSize: 13, color: Colors.black87),
            ),
            SizedBox(height: 20),
            TextField(
              enabled: false,
              decoration: InputDecoration(
                hintText: "https://bricoram.com/p/172528",
                filled: true,
                fillColor: Colors.grey.shade200,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(8),
                  borderSide: BorderSide.none,
                ),
                contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
              ),
            ),
            SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: TextButton.icon(
                onPressed: () {
                  // Add your clipboard copy logic here
                   Clipboard.setData(ClipboardData(text: "https://bricoram.com/p/172528"));
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Lien copié !')),
                  );
                },
                icon: Icon(Icons.copy, color: Colors.grey[700], size: 20),
                label: Text(
                  "Copier le lien",
                  style: TextStyle(color: Colors.grey[700]),
                ),
                style: TextButton.styleFrom(
                  foregroundColor: Colors.grey[700],
                  padding: EdgeInsets.symmetric(horizontal: 0, vertical: 0),
                  textStyle: TextStyle(fontSize: 15),
                ),
              ),
            ),
            SizedBox(height: 25),
            // Notifications
            Row(
              children: [
                Icon(Icons.notifications_none, color: Colors.black87),
                SizedBox(width: 6),
                Text(
                  "Notifications",
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ],
            ),
            SizedBox(height: 4),
            Text(
              "Comment souhaitez-vous recevoir vos notification?",
              style: TextStyle(fontSize: 13, color: Colors.black87),
            ),
            SizedBox(height: 25),
            Row(
              children: [
                Expanded(
                  child: GestureDetector(
                    onTap: () => setState(() => selectedNotif = 1),
                    child: Container(
                      padding: EdgeInsets.symmetric(vertical: 14),
                      decoration: BoxDecoration(
                        color: selectedNotif == 1 ? green : Colors.white,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: selectedNotif == 1 ? green : Colors.grey.shade300,
                          width: 1.5,
                        ),
                      ),
                      child: Column(
                        children: [
                          Icon(Icons.phone_android,
                              color: selectedNotif == 1 ? Colors.white : green),
                          SizedBox(height: 4),
                          Text(
                            "Application\nUniquement",
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: selectedNotif == 1 ? Colors.white : Colors.black87,
                              fontWeight: FontWeight.w500,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(width: 10),
                Expanded(
                  child: GestureDetector(
                    onTap: () => setState(() => selectedNotif = 2),
                    child: Container(
                      padding: EdgeInsets.symmetric(vertical: 14),
                      decoration: BoxDecoration(
                        color: selectedNotif == 2 ? green : Colors.white,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: selectedNotif == 2 ? green : Colors.grey.shade300,
                          width: 1.5,
                        ),
                      ),
                      child: Column(
                        children: [
                          Icon(Icons.email,
                              color: selectedNotif == 2 ? Colors.white : green),
                          SizedBox(height: 4),
                          Text(
                            "Application +\nE-Mail",
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: selectedNotif == 2 ? Colors.white : Colors.black87,
                              fontWeight: FontWeight.w500,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 50),
            Center(
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => SignInPage()),
                      );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red.shade100,
                    foregroundColor: Colors.red,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    padding: EdgeInsets.symmetric(vertical: 18),
                  ),
                  child: Text(
                    "Se deconnecter",
                    style: TextStyle(
                      color: Colors.red,
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                    ),
                  ),
                ),
              ),
            ),
            SizedBox(height: 16),
          ],
        ),
      ),
    );
  }
}