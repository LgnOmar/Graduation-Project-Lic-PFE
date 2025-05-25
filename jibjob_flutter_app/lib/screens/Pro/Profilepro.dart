import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:auto_size_text/auto_size_text.dart';
import 'package:jibjob/screens/ListeChoix.dart';
import 'package:jibjob/screens/MesPrix.dart';
import 'dart:io';
import 'package:jibjob/Pro.dart';
import 'package:jibjob/screens/ParametresPro.dart';


class Profilepro extends StatefulWidget {

  int Pro_Position ;

  Profilepro({
    required this.Pro_Position,
  }) ;


  @override
  _ProfileproState createState() => _ProfileproState();
}



class _ProfileproState extends State<Profilepro> {


  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366);

  int selectedNavIndex = 3;

  @override
  Widget build(BuildContext context) {
    Pro pro = Liste_Pros[widget.Pro_Position] ;

    print ("Nb = ${Pro.nb}") ;
    
    final double screenHeight = MediaQuery.of(context).size.height;
    final double screenWidth = MediaQuery.of(context).size.width;
    
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
              Navigator.push(context, MaterialPageRoute(builder: (_) => ListeChoix(Pro_Position : widget.Pro_Position,)));
             
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
        onPressed: () {},
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
      body: SingleChildScrollView( 
      child : Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(height: 50),
            Text("Profile" , style : TextStyle(fontSize: 30 , color : Color.fromRGBO(19, 1, 96, 1))) ,
            SizedBox(height: 20),
            // Profile Section

            Container(
              margin :EdgeInsets.fromLTRB(screenWidth * 0.02, 0, screenWidth * 0.02, 0),
              decoration: BoxDecoration(
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
              child: Column(

              children: [
                Row(
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
                    child: pro.image == null ?
                    Image.asset('assets/_Unkown.png' , fit: BoxFit.cover , width: double.infinity, height: double.infinity,) :
                    Image.file(File(pro.image!,), fit: BoxFit.cover , width: double.infinity, height: double.infinity,) ,
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
                          pro.name == null ? "" : pro.name!,
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
                          pro.address == null ? "" : pro.address!,
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
                          pro.phone == null ? "" : pro.phone!,
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
            
            
            SizedBox(height: 35) ,

              
              Container(
    width: double.infinity,
    child: Wrap(
        alignment: WrapAlignment.start,
        runAlignment: WrapAlignment.start,
        spacing: 8,
        runSpacing: 5,
        children: 
                 pro.Services.map((service) {
            return Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
                decoration: ShapeDecoration(
                    color: const Color(0xFFE7E7E7),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
                ),
                child: Row(
                    mainAxisSize: MainAxisSize.min,
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.center,
                    spacing: 10,
                    children: [
                        Text(
                            service ?? '',
                            style: TextStyle(
                                color: Colors.black,
                                fontSize: 11,
                                fontFamily: 'DM Sans',
                                fontWeight: FontWeight.w600,
                                height: 0.95,
                            ),
                        ),
                    ],
                ),
            );
                 }).toList() ,))
                 
                 ]))  ,


            
SizedBox(height: 32),


Row(
              children: [
                                GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => Mesprix(Pro_Position :widget.Pro_Position)));
                          },
                child :
                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
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
                              Image.asset('assets/profile.png', width: 50, height: 50 ,),
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
                    ))
                            ],
                          ))),

                GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => Mesprix(Pro_Position :widget.Pro_Position)));
                          },
                child :
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
                              Image.asset('assets/prix.png', width: 50, height: 50 ,),
                              Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                      "Mes Prix",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: darkPurple,
                        fontSize: screenHeight * 0.025 ,
                      ),
                      maxLines: 1,  
                      minFontSize: 1,
                    ))
                            ],
                          )))
              ],
            ),
            SizedBox(height: 20) ,
            Row(
              children: [
                                GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => Mesprix(Pro_Position :widget.Pro_Position)));
                          } ,
                child :
                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
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
                              Image.asset('assets/social.png', width: 50, height: 50 ,),
                              Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                      "Réseaux Sociaux",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: darkPurple,
                        fontSize: screenHeight * 0.025 ,
                      ),
                      maxLines: 1,  
                      minFontSize: 1,
                    ))
                            ],
                          ))) ,

                                          GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => Mesprix(Pro_Position :widget.Pro_Position)));
                          },
                child :
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
                              Image.asset('assets/points.png', width: 50, height: 50 ,),
                              Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                      "Mes Points",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: darkPurple,
                        fontSize: screenHeight * 0.025 ,
                      ),
                      maxLines: 1,  
                      minFontSize: 1,
                    ))
                            ],
                          )))                
              ],
            ),

            SizedBox(height: 20) ,
            Row(
              children: [
                GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => Mesprix(Pro_Position :widget.Pro_Position)));
                          },
                child :
                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
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
                              Image.asset('assets/accueil.png', width: 50, height: 50 ,),
                              Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                      "Acceuil",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: darkPurple,
                        fontSize: screenHeight * 0.025 ,
                      ),
                      maxLines: 1,  
                      minFontSize: 1,
                    ))
                            ],
                          ))) ,

                          GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => ParametresPro()));
                          },
                child :
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
                              Image.asset('assets/parametres.png', width: 50, height: 50 ,),
                              Container(
                    width: screenWidth * 0.35,
                    height: screenHeight * 0.04,
                  child : AutoSizeText(
                      "Paramétres",
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: darkPurple,
                        fontSize: screenHeight * 0.025 ,
                      ),
                      maxLines: 1,  
                      minFontSize: 1,
                    ))
                            ],
                          )))
              ],
            ),

            SizedBox(height: 20) ,
            Row(
              children: [
                GestureDetector(
                          onTap: () {
                            Navigator.push(context, MaterialPageRoute(builder: (_) => Mesprix(Pro_Position :widget.Pro_Position)));
                          },
                child :
                Container( 
                  padding: EdgeInsets.fromLTRB(5, 0, 0, 0),
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
                              Image.asset('assets/aide.png', width: 50, height: 50 ,),
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
                    ))
                            ],
                          )))
          ])
        ,]
      ),
      )
    )
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