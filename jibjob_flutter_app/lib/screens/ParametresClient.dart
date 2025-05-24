import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:jibjob/screens/SignInPage.dart';

class ParametresClient extends StatefulWidget {
  @override
  State<ParametresClient> createState() => _ParametresClientState();
}

class _ParametresClientState extends State<ParametresClient> {
  int selectedNotif = 1; // 1: Application, 2: Application + Email

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