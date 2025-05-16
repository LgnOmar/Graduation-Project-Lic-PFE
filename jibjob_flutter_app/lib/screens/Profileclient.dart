import 'package:flutter/material.dart';

// ignore: must_be_immutable
class Profileclient extends StatelessWidget {

  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366); // WhatsApp green

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[300] ,
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
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(height: 50),
            Text("Profile" , style : TextStyle(fontSize: 30 , color : Color.fromRGBO(19, 1, 96, 1))) ,
            SizedBox(height: 20),
            // Profile Section
            Container(
              margin :EdgeInsets.fromLTRB(10, 0, 10, 0),
              height: 150,
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 8,
                    spreadRadius: 4,
                  ),
                ],
              ),
              padding: EdgeInsets.all(16),
              child: Row(
                children: [
                  // Profile Image and Info
                  CircleAvatar(
                    radius: 60,
                    backgroundImage: AssetImage('assets/profilePhoto.png'),
                  ),
                  SizedBox(width: 50),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start ,
                  children : [Text(
                    'Omar' ,
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w600 ,
                      color: darkPurple,
                    ),
                  ),
                  SizedBox(height: 4),
                  
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
                    decoration: ShapeDecoration(
                    color: const Color(0xFFE7E7E7),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
                      ),
                  child :Text(
                    'Baraki, Alger' ,
                    style: TextStyle(fontSize: 14, color: Colors.black54),
                  )),
                  SizedBox(height: 8),
                  // Phone number with WhatsApp icon
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6) , 
                        decoration : ShapeDecoration(
                          color: Colors.greenAccent[400],
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
                        ),

                      child :  Row(children: [
                      Icon(Icons.call, color: Colors.white),
                      SizedBox(width: 8),
                      Text(
                        '0598329037' ,
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: darkPurple,
                        ),
                      )])),
                    ],
                  )])
                ],
              ),
            ),
            SizedBox(height: 32),

            // Action Buttons (Mon Profile, Accueil, Parametres, Aide et Support)
            Row(
              //mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
                  margin: EdgeInsets.fromLTRB(10, 0, 0, 0),
                  width: 180,
                  height: 115,
                  alignment: Alignment.topLeft ,
                decoration: ShapeDecoration(
                    color: Colors.white ,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ), 
                child : 
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start ,
                  children: [
                  IconButton(
                    icon: Image.asset('assets/profile.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
                  Text("Mon Profile" , style : TextStyle(fontSize: 20 , fontWeight: FontWeight.w500 , fontFamily: 'DM Sans'))
                  ])

                )
                ,

                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
                  margin: EdgeInsets.fromLTRB(10, 0, 0, 0),
                  width: 180,
                  height: 115,
                  alignment: Alignment.topLeft ,
                decoration: ShapeDecoration(

                    color: Colors.white ,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ), 
                child :
                
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start ,
                  children: [
                  IconButton(
                    icon: Image.asset('assets/accueil.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
                  Text("Acceuil" , style : TextStyle(fontSize: 20 , fontWeight: FontWeight.w500 , fontFamily: 'DM Sans'))
                  ])
                )
              ],
            ),
            SizedBox(height: 20) ,
            Row(
              //mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
                  margin: EdgeInsets.fromLTRB(10, 0, 0, 0),
                  width: 180,
                  height: 115,
                  alignment: Alignment.topLeft ,
                decoration: ShapeDecoration(

                    color: Colors.white ,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ), 
                child : Column( crossAxisAlignment: CrossAxisAlignment.start ,
                  children: [
                  IconButton(
                    icon: Image.asset('assets/parametres.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
                  Text("Parametres" , style : TextStyle(fontSize: 20 , fontWeight: FontWeight.w500 , fontFamily: 'DM Sans'))
                  ])
                ),

                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
                  margin: EdgeInsets.fromLTRB(10, 0, 0, 0),
                  width: 180,
                  height: 115,
                decoration: ShapeDecoration(

                    color: Colors.white ,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ), 
                child : Column(
                  crossAxisAlignment: CrossAxisAlignment.start ,
                  children: [
                  IconButton(
                    icon: Image.asset('assets/aide.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
                  Text("Aide et Support" , style : TextStyle(fontSize: 19 , fontWeight: FontWeight.w500 , fontFamily: 'DM Sans'))
                  ])
                )
                
                
              ],
            ),
            SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}

class ActionButton extends StatelessWidget {
  final IconData icon;
  final String text;

  ActionButton({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        CircleAvatar(
          radius: 30,
          backgroundColor: Colors.grey[200],
          child: Icon(icon, color: Colors.red, size: 30),
        ),
        SizedBox(height: 8),
        Text(
          text ,
          style: TextStyle(
            fontSize: 14,
            color: Colors.red,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}
