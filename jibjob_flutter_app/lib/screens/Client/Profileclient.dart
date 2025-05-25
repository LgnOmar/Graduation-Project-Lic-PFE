import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:auto_size_text/auto_size_text.dart';
import 'dart:io';
import 'package:jibjob/Person.dart';
import 'package:jibjob/screens/Client/MakeOrder.dart';
import 'package:jibjob/screens/Client/NewOrder.dart';
import 'package:jibjob/screens/ParametresClient.dart';

class Profileclient extends StatefulWidget {

  int Client_Position ;

  Profileclient({
    required this.Client_Position,
  }) ;


  @override
  _ProfileclientState createState() => _ProfileclientState();
}


class _ProfileclientState extends State<Profileclient> {

  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366);

  int selectedNavIndex = 3;


  @override
  Widget build(BuildContext context) {
    double screenHeight = MediaQuery.of(context).size.height;
    double screenWidth = MediaQuery.of(context).size.width ;
    return Scaffold(
      backgroundColor: Colors.grey[300] ,

      bottomNavigationBar : Stack(
  alignment: Alignment.center,
  children: [
    Container(
      height: 95,
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 8,
            offset: Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _NavBarItem(
            icon: Icons.groups,
            label: "Pros",
            selected: selectedNavIndex == 0,
            onTap: () => setState(() => selectedNavIndex = 0),
          ),
          _NavBarItem(
            icon: Icons.campaign,
            label: "Demandes",
            selected: selectedNavIndex == 1,
            onTap: () => setState(() {
              selectedNavIndex = 1 ;
              Navigator.push(context, MaterialPageRoute(builder: (_) => MakeOrder(Client_Position : widget.Client_Position,)));
              }),
          ),
          SizedBox(width: 56), // Space for the FAB
          _NavBarItem(
            icon: Icons.message,
            label: "Messages",
            selected: selectedNavIndex == 2,
            onTap: () => setState(() => selectedNavIndex = 2),
          ),
          _NavBarItem(
            icon: Icons.person,
            label: "Compte",
            selected: selectedNavIndex == 3,
            onTap: () => setState(
              () {
                selectedNavIndex = 3;
              }),
          ),
        ],
      ),
    ),
    Positioned(
  bottom: 20,
  child: ClipOval(
    child: Container(
      width: 60,
      height: 60,
      child: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => NewOrder(),
            ),
          );
        },
        backgroundColor: Color(0xFF20004E),
        elevation: 4,
        child: Icon(Icons.add, size: 40),
      ),
    ),
  ),
)

,
  ],
),
      body: Padding(
        padding: EdgeInsets.all(screenWidth * 0.03),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(height: 50),
            Text("Profile" , style : TextStyle(fontSize: screenHeight * 0.035 , color : Color.fromRGBO(19, 1, 96, 1))) ,
            SizedBox(height: 20),
            // Profile Section
            Container(
              margin :EdgeInsets.fromLTRB(screenWidth * 0.02, 0, screenWidth * 0.02, 0),
              height: screenHeight * 0.15,
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
              padding: EdgeInsets.all(screenWidth * 0.02),
              child: Row(
                children: [
                  // Profile Image and Info
                  Container(
                    height : screenWidth * 0.25 ,
                    width: screenWidth * 0.25, 
                    decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(1000),
                                border: Border.all(
                                  color: Color(0xFF130160),
                                  width: 2,
                                ), 
                              ),

                    child : ClipOval(
                    child: Liste_Clients[widget.Client_Position].image == null ?
                    Image.asset('assets/_Unkown.png' , fit: BoxFit.cover , width: double.infinity, height: double.infinity,) :
                    Image.file(File(Liste_Clients[widget.Client_Position].image! ,), fit: BoxFit.cover , width: double.infinity, height: double.infinity,) ,
                    ) ,
                    
                  ) ,
                  SizedBox(width: screenWidth * 0.05),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start ,
                  children : [
                    Container(
                      padding: EdgeInsets.fromLTRB(screenWidth * 0.02, 0, screenWidth * 0.02, 0),
                      width: screenWidth * 0.54,
                      height: screenHeight * 0.035,
                      child :
                              AutoSizeText(
                          Liste_Clients[widget.Client_Position].name == null ? "null" : Liste_Clients[widget.Client_Position].name!,
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.02 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        ),
                      ),


                  SizedBox(height: 4),
                  
                  Container(
                    width: screenWidth * 0.54,
                    height: screenHeight * 0.035,
                    padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
                    decoration: ShapeDecoration(
                    color: const Color(0xFFE7E7E7),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
                      ),
                  child : AutoSizeText(
                          Liste_Clients[widget.Client_Position].address == null ? "null" : Liste_Clients[widget.Client_Position].address!,
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.02 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        )) ,
                  SizedBox(height: 8),
                  // Phone number with WhatsApp icon
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        width: screenWidth * 0.54 ,
                        height: screenHeight * 0.04,
                        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6) , 
                        decoration : ShapeDecoration(
                          color: Colors.greenAccent[400],
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
                        ),

                      child :  Row(children: [
                      Icon(Icons.call, color: Colors.white),
                      SizedBox(width: 8),
                      Container(
                        child :AutoSizeText(
                          Liste_Clients[widget.Client_Position].phone == null ? "null" : Liste_Clients[widget.Client_Position].phone!,
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.025 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        ))])),
                    ],
                  )])
                ],
              ),
            ),
            SizedBox(height: 32),
            Row(
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

                  Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                          "Mon Profile",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.025 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        )
                  ) 
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
                  Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                          "Accueil",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.025 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        )
                  ) 
                  ])
                )
              ],
            ),
            SizedBox(height: 20) ,
            Row(
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
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => ParametresClient()),
                      );
                    },
                  ),
                  Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                          "Parametres",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.025 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        )
                  ) 
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
                  Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                          "Aide & Support",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            color: darkPurple,
                            fontSize: screenHeight * 0.025 ,
                          ),
                          maxLines: 1,  
                          minFontSize: 1,
                        )
                  ) 
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




class _NavBarItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool selected;
  final VoidCallback? onTap;

  const _NavBarItem({
    required this.icon,
    required this.label,
    this.selected = false,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: selected ? Color(0xFF20004E) : Colors.deepPurple.shade100,
          ),
          SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              color: selected ? Color(0xFF20004E) : Colors.deepPurple.shade100,
              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }
}