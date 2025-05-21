import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:auto_size_text/auto_size_text.dart';
import 'ClientSignUpPage.dart';
import 'dart:io';

// ignore: must_be_immutable
class Profileclient extends StatelessWidget {

  final Color darkPurple = Color(0xFF20004E);
  final Color green = Color(0xFF25D366);

  TextEditingController emailController ;
  TextEditingController passwordController;
  TextEditingController nameController;
  TextEditingController phoneController;
  TextEditingController cityController;
  TextEditingController presentationController ;
  XFile? imagePath ;


  Profileclient(
    this.emailController ,
    this.passwordController,
    this.nameController,
    this.phoneController,
    this.cityController,
    this.presentationController ,
    this.imagePath ,
    );



  @override
  Widget build(BuildContext context) {
    double screenHeight = MediaQuery.of(context).size.height;
    double screenWidth = MediaQuery.of(context).size.width ;
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
                    child: imagePath == null ?
                    Image.asset('assets/_Unkown.png' , fit: BoxFit.cover , width: double.infinity, height: double.infinity,) :
                    Image.file(File(imagePath!.path ,), fit: BoxFit.cover , width: double.infinity, height: double.infinity,) ,
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
                          nameController.text,
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
                          cityController.text ,
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
                          phoneController.text,
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
                    onPressed: () {},
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
