import 'package:auto_size_text/auto_size_text.dart';
import 'package:flutter/material.dart';
import 'package:jibjob/screens/ListeChoix.dart';
import 'ProSignUpPage0.dart';
import 'Profilepro.dart' ;
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'dart:ui';
import 'package:jibjob/Pro.dart';
import 'ProSignUpPage0.dart';


class ProSignUpPage extends StatefulWidget {
  Pro person ;

  
  ProSignUpPage({
    required this.person,
  }) ;
  

  @override
  _ProSignUpPageState createState() => _ProSignUpPageState();
}

class _ProSignUpPageState extends State<ProSignUpPage> {


  final Color darkPurple = Color(0xFF20004E);
  final presentationController = TextEditingController();

  final List<Map<String, dynamic>> categories = [
    {
      "name": "Maison",
      "services": [
        {"name": "Plombier", "image": "assets/plombier.png" },
        {"name" : "Peintre" , "image": "assets/peintre.png"},
        {"name" : "Électricien" , "image": "assets/electricien.png"}
      ]
    },
    {
      "name": "Auto",
      "services": [
        {"name" : "Mécanicien", "image": "assets/mecanicien.png"},
        {"name" : "Tolier", "image": "assets/tolier.png"},
        {"name" : "Lavage", "image": "assets/lavage.png"}
      ]
    },
    {
      "name": "Beauté",
      "services": [
        {"name" : "Coiffeur", "image": "assets/coiffeur.png"},
        {"name" : "Esthéticienne", "image": "assets/estheticienne.png"},
        {"name" : "Maquilleur", "image": "assets/maquilleur.png"}
      ]
    },
    {
      "name": "Informatique",
      "services": [
        {"name" : "Développeur", "image": "assets/developeur.png"},
        {"name" : "Réparateur", "image": "assets/reparation.png"},
        {"name" : "Technicien réseau", "image": "assets/technicien.png"}
      ]
    },
    {
      "name": "Santé",
      "services": [
        {"name" : "Infirmier", "image": "assets/Infirmier.png"},
        {"name" : "Coach Kinésithérapeute", "image": "assets/Kinesitherapeute.png"},
        {"name" : "Pharmacien", "image": "assets/Pharmacien.png"}
      ]
    },
    {
      "name": "Bâtiment",
      "services": [
        {"name" : "Maçon", "image": "assets/Macon.png"},
        {"name" : "Carreleur", "image": "assets/Carreleur.png"},
        {"name" : "Charpentier", "image": "assets/Charpentier.png"}
      ]
    },
    {
      "name": "Transports",
      "services": [
        {"name" : "poids lourd", "image": "assets/poids_lourd.png"},
        {"name" : "Livreur", "image": "assets/Livreur.png"},
        {"name" : "Taxi", "image": "assets/taxi.png"}
      ]
    },
    {
      "name": "Éducation",
      "services": [
        {"name" : "Professeur", "image": "assets/educateur specialise.png"},
        {"name" : "auto-école", "image": "assets/auto-ecole.png"},
        {"name" : "Éducateur spécialisé", "image": "assets/Professeur.png"}
      ]
    },
  ];

  List<String?> selectedServices = [] ;

final ImagePicker _picker = ImagePicker();

  
  XFile? _image;
  XFile? imageController ;
  List<XFile?> _images = [null];
  List<String?> imagesController = [null] ;

  int number = 1 ;

void AddPicture() {
    setState(() {
      if(_images[number-1] != null) {
        _images.add(null); 
        imagesController.add(null);
        number++; 
      }
      
    });
  }

  void RemovePicture() {
    if (number > 1) {
      setState(() {
        _images.removeLast();
        imagesController.removeLast();
        number--; 
      });
    }else if (number == 1) {
      _images[0] = null ;
      if (imagesController.isNotEmpty) {
        imagesController[0] = null;
      }
    }
  }



  // Function to pick the profile image
  Future<void> _pickImageProfile() async {
    try {
      // Pick an image from the gallery
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);

      if (image != null) {
        setState(() {
          _image = image;
          imageController = _image ;
        });
      } else {
        // Handle case where user cancels the picker
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('No image selected.')),
        );
      }
    } catch (e) {
      // Handle any errors that occur
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to pick image: $e')),
      );
    }
  }

  Future<void> _pickImage(int index) async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
      if (image != null) {
        setState(() {
          _images[index] = image;
          while (imagesController.length <= index) {
          imagesController.add(null);
        }
          imagesController[index] = _images[index]?.path ;
        });
      } else {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('No image selected.')));
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to pick image: $e')));
    }
  }



  @override
  Widget build(BuildContext context) {

    

    double screenHeight = MediaQuery.of(context).size.height;
    double screenWidth = MediaQuery.of(context).size.width;




    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        foregroundColor: Color(0xFF130160),
        elevation: 0,
        leading: IconButton(
      icon: Icon(Icons.arrow_back),
      onPressed: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => ProSignUpPage0(),
          ),
        );
      },
    ),
        title: Text("Créer un compte pro", style: TextStyle(color: Color(0xFF130160),)),
        centerTitle: true,
      ),
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: screenWidth * 0.05),
          child: ListView(
            children: [
              SizedBox(height: 24),

              buildField(

                "Presentation",
               "ex: j'ai x années d’expérience en x. Le peux fournir le service a...",
                presentationController,
                subLabel: "Presenter vous en quelques mots aux clients",
                ),

              SizedBox(height: 24),

              // Photo de profile
              Text("Photo de Profile",
                  style: TextStyle(fontWeight: FontWeight.bold)),
              Text("photo de votre visage"),

              SizedBox(height: 12),

              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            SizedBox(height: screenHeight * 0.03),
                            Container(
                              height: screenHeight * 0.2,
                              width: screenHeight * 0.2,
                              decoration: BoxDecoration(
                                color: Colors.grey[350] ,
                                borderRadius: BorderRadius.circular(1000),
                                border: Border.all(
                                  color: Color(0xFF130160),
                                  width: 2,
                                ), 
                              ),

                              child: GestureDetector(
                                onTap: _pickImageProfile,
                                child: _image == null
                                    ? ClipOval(
                                      child: Image.asset(
                                        "assets/Picture_Icon.png",
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      ))
                                    : ClipOval(
                                      child: Image.file(
                                        File(_image!.path),
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      )),
                              ),
                            
                            )
                          ],
                        ),
              SizedBox(height: 24),

              // Realisations
              Text("Realisations",
                  style: TextStyle(fontWeight: FontWeight.bold)),
              Text("Montrez aux clients des photos de vos travaux"),
              SizedBox(height: 12),

              SizedBox(height: 12),

              Wrap(
                
                spacing: 8,
                runSpacing: 8,
                children: List.generate(
                  number,
                      (index) => Container(
                              height: screenHeight * 0.1,
                              width: screenHeight * 0.1,
                              decoration: BoxDecoration(
                                color: Colors.grey[350] ,
                                borderRadius: BorderRadius.circular(1000),
                                border: Border.all(
                                  color: Color(0xFF130160),
                                  width: 2,
                                ), 
                              ),

                              child: GestureDetector(
                                onTap: () => _pickImage(index),
                                child: _images[index] == null
                                    ? ClipOval(
                                      child: Image.asset(
                                        "assets/Picture_Icon.png",
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      ))
                                    : ClipOval(
                                      child: Image.file(
                                        File(_images[index]!.path),
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: double.infinity,
                                      )),
                              ),
                            
                            )
                ) , 

              ),

              SizedBox(height: 24),

              Row(
              
              children: [
              
               Container(
                width: screenWidth*0.4,
               child : ElevatedButton(
                onPressed: AddPicture,

                child: AutoSizeText('Ajouter une photo' , maxLines: 1, minFontSize: 1,) ,
              ),
               ),

                SizedBox(width: screenWidth*0.1,),


               Container(
                width: screenWidth*0.4,
               child : ElevatedButton(
                onPressed: RemovePicture,
                child: AutoSizeText('Supprimer une photo' , maxLines: 1, minFontSize: 1,),
              ),
               )
               
               
               
               ]) ,

              SizedBox(height: 24),

              // Services

              ...categories.map((category) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(height: 24),
                    Row(
                      children: [
                        Icon(Icons.category, color: darkPurple, size: 18),
                        SizedBox(width: 6),
                        Text(category["name"]),
                      ],
                    ),
                    SizedBox(height: 12),
                    Wrap(
                      spacing: 12,
                      runSpacing: 12,
                      children: category["services"].map<Widget>((service) {
                        String name = service["name"];
                        String imagePath = service["image"];
                        bool isSelected = selectedServices.contains(name);

                        return GestureDetector(
                          onTap: () {
                            setState(() {
                              if (isSelected) {
                                selectedServices.remove(name);
                              } else {
                                selectedServices.add(name);
                              }
                            });
                          },
                          child: Column(
                            children: [
                              CircleAvatar(
                                radius: screenWidth * 0.25 / 2,
                                backgroundColor: isSelected ? Colors.deepPurple : Colors.grey[300],
                                backgroundImage:  isSelected ? null : AssetImage(imagePath),
                                
                                child: isSelected
                                    ? Icon(Icons.check, color: Colors.white, size: 32)
                                    : null,
                              ),
                              SizedBox(height: 6),
                              Text(
                                name,
                                textAlign: TextAlign.center,
                                style: TextStyle(fontSize: 12),
                              ),
                            ],
                          ),
                        );
                      }).toList(),
                    ),
                  ],
                );
              }).toList(),
              SizedBox(height: 32),

              ElevatedButton(
                onPressed: () {
                  widget.person.description = presentationController.text ;
                  widget.person.image = imageController?.path ;
                  widget.person.images = imagesController ;
                  widget.person.Services = selectedServices ;

                  Liste_Pros.add(widget.person);

                  Navigator.push(context, MaterialPageRoute(builder: (_) => ListeChoix(Pro_Position : Pro.nb -1) ));
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: darkPurple,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(24),
                  ),
                  padding: EdgeInsets.symmetric(vertical: 14),
                ),
                child: Text("Terminer" , style: TextStyle(fontSize: 16, color: Colors.white, fontWeight: FontWeight.bold) ,),
              ),
              SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

    Widget buildField(String label, String hint, TextEditingController controller,
      {String? subLabel, bool isPassword = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label , style: TextStyle(fontSize: 18)),
          if (subLabel != null)
            Padding(
              padding: const EdgeInsets.only(top: 2.0),
              child: Text(
                subLabel,
                style: TextStyle(fontSize: 12, color: Colors.black54),
              ),
            ),
          SizedBox(height: 6),
          TextField(
            maxLines: 4,
            controller: controller,
            obscureText: isPassword,
            decoration: InputDecoration(
              hintText: hint,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              contentPadding: EdgeInsets.symmetric(horizontal: 12),
              fillColor: Colors.grey[200] ,
              filled: true ,
            ),
          ),
        ],
      ),
    );
  }
}