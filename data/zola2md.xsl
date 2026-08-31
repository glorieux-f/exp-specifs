<?xml version="1.0" encoding="UTF-8"?>
<xsl:transform version="1.1" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns="http://www.w3.org/1999/xhtml" xmlns:html="http://www.w3.org/1999/xhtml"
  xmlns:tei="http://www.tei-c.org/ns/1.0" 
  xmlns:data="urn:data" 
  xmlns:exsl="http://exslt.org/common"
  
  exclude-result-prefixes="tei html data"
  extension-element-prefixes="exsl"
  >
  <!-- 
  -->
  <xsl:include href="../../teinte-xsl/tei_txt/tei_markdown.xsl"/>
  <xsl:param name="filename"/>
  <xsl:param name="outdir"/>
  
  <xsl:variable name="yamline">
    <xsl:text>---</xsl:text>
    <xsl:value-of select="$lf"/>
  </xsl:variable>
  <xsl:variable name="bookid" select="substring-before($filename, '_')"/>
  <xsl:template match="/">
    <xsl:variable name="chapcount" select="count(/tei:TEI/tei:text/tei:body//tei:div[@type='chapter'])"/>
    <xsl:choose>
      <xsl:when test="$chapcount = 1">###error</xsl:when>
    </xsl:choose>
    <xsl:value-of select="$lf"/>
    <xsl:choose>
      <xsl:when test="$chapcount = 0">
        <xsl:variable name="href" select="concat($outdir, $bookid, '.md')"/>
        <xsl:value-of select="$href"/>
        <xsl:value-of select="$lf"/>
        <exsl:document href="{$href}" method="text" omit-xml-declaration="yes" encoding="UTF-8" indent="yes">
          <xsl:value-of select="$yamline"/>
          <xsl:text>identifier: </xsl:text>
          <xsl:value-of select="$bookid"/>
          <xsl:value-of select="$lf"/>
          <xsl:text>creator: Balzac, Honoré de</xsl:text>
          <xsl:value-of select="$lf"/>
          <xsl:text>created: </xsl:text>
          <xsl:value-of select="$docdate"/>
          <xsl:value-of select="$lf"/>
          <xsl:text>title: </xsl:text>
          <xsl:value-of select="$doctitle"/>
          <xsl:value-of select="$lf"/>
          <xsl:value-of select="$yamline"/>
          <xsl:value-of select="$lf"/>
          <xsl:apply-templates select="/tei:TEI/tei:text/tei:body/node()" mode="md"/>
        </exsl:document>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="/tei:TEI/tei:text/tei:body//tei:div[@type='chapter']" mode="md"
        />
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>


  <xsl:template match="tei:div[@type = 'chapter']" mode="md">
    <xsl:variable name="no">
      <xsl:number count="tei:div[@type = 'chapter']" format="01" from="tei:body" level="any"/>
    </xsl:variable>
    <xsl:variable name="docid">
      <xsl:value-of select="$bookid"/>
      <xsl:text>-</xsl:text>
      <xsl:value-of select="$no"/>
    </xsl:variable>
    <xsl:variable name="href" select="concat($outdir, $docid, '.md')"/>
    <xsl:variable name="title">
      <xsl:apply-templates select="tei:head" mode="title"/>
    </xsl:variable>
    <xsl:value-of select="$href"/>
    <xsl:value-of select="$lf"/>
    <exsl:document href="{$href}" method="text" omit-xml-declaration="yes" encoding="UTF-8" indent="yes">
      <xsl:value-of select="$yamline"/>
      <xsl:text>identifier: </xsl:text>
      <xsl:value-of select="$docid"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>creator: </xsl:text>
      <xsl:value-of select="$byline"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>created: </xsl:text>
      <xsl:value-of select="$docdate"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>isPartOf: </xsl:text>
      <xsl:value-of select="$doctitle"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>title: </xsl:text>
      <xsl:value-of select="$title"/>
      <xsl:value-of select="$lf"/>
      <xsl:value-of select="$yamline"/>
      <xsl:value-of select="$lf"/>
      <xsl:apply-templates mode="md"/>
    </exsl:document>
  </xsl:template>
  <xsl:template match="tei:note" mode="md"/>
  <xsl:template match="tei:hi" mode="md">
    <xsl:apply-templates/>
  </xsl:template>
</xsl:transform>
