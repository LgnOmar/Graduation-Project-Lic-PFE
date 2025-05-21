import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:auto_size_text/auto_size_text.dart';
import 'dart:io';
import 'SplashScreen.dart';

// ignore: must_be_immutable
class Profilepro extends StatelessWidget {

     String? email ;
    String? password ;
    String? name ;
    String? phone ;
    String? city ;
    String? presentation ;
    Set<String> selectedServices = {};
    XFile? imageFile ;
    String? imagePath ;
    List<XFile?> _images = [null];

    TextEditingController emailController = TextEditingController();

    Profilepro(this.email , this.password , this.name , this.phone , this.city , this.presentation ,  this.selectedServices , this.imageFile , _images ) {}


  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366); // WhatsApp green

  @override
  Widget build(BuildContext context) {

    imagePath = imageFile?.path;
    
    final double screenHeight = MediaQuery.of(context).size.height;
    final double screenWidth = MediaQuery.of(context).size.width;
    
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
                    child: imageFile == null ?
                    Image.asset('assets/_Unkown.png' , fit: BoxFit.cover , width: double.infinity, height: double.infinity,) :
                    Image.file(File(imageFile!.path ,), fit: BoxFit.cover , width: double.infinity, height: double.infinity,) ,
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
                          name!,
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
                          city!,
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
                          phone!,
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
                 selectedServices.map((service) {
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
                            service,
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
                    icon: Image.asset('assets/prix.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
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
                    icon: Image.asset('assets/social.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
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
                    icon: Image.asset('assets/points.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
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
                    icon: Image.asset('assets/accueil.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
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
                    icon: Image.asset('assets/parametres.png', width: 50, height: 50 ,),
                    onPressed: () {},
                  ),
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
                ),

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
