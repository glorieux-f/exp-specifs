<?xml version="1.0" encoding="UTF-8"?>
<xsl:transform version="1.1" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns="http://www.w3.org/1999/xhtml" xmlns:html="http://www.w3.org/1999/xhtml"
  xmlns:tei="http://www.tei-c.org/ns/1.0" 
  xmlns:data="urn:data" 
  xmlns:exsl="http://exslt.org/common"
  exclude-result-prefixes="tei html data exsl">
  <xsl:include href="../../teinte-xsl/tei_txt/tei_markdown.xsl"/>
  <xsl:param name="filename"/>
  <xsl:param name="outdir"/>
  <xsl:variable name="bookno">
    <xsl:value-of select="substring-before(substring-after($filename, 'balzac-'), '-FC')"/>
  </xsl:variable>
  <data:dates>
    <data:item no="00">1842</data:item>
    <data:item no="01">1830</data:item>
    <data:item no="02">1830</data:item>
    <data:item no="03">1841</data:item>
    <data:item no="04">1832</data:item>
    <data:item no="05">1844</data:item>
    <data:item no="06">1844</data:item>
    <data:item no="07">1842</data:item>
    <data:item no="08">1830</data:item>
    <data:item no="09">1830</data:item>
    <data:item no="10">1830</data:item>
    <data:item no="11">1832</data:item>
    <data:item no="12">1830</data:item>
    <data:item no="13">1842</data:item>
    <data:item no="14">1838</data:item>
    <data:item no="15">1833</data:item>
    <data:item no="16">1833</data:item>
    <data:item no="17">1833</data:item>
    <data:item no="18">1843</data:item>
    <data:item no="19">1839</data:item>
    <data:item no="20">1830</data:item>
    <data:item no="21">1834</data:item>
    <data:item no="22">1835</data:item>
    <data:item no="23">1832</data:item>
    <data:item no="24">1839</data:item>
    <data:item no="25">1837</data:item>
    <data:item no="26">1836</data:item>
    <data:item no="27">1835</data:item>
    <data:item no="28">1831</data:item>
    <data:item no="29">1841</data:item>
    <data:item no="30">1833</data:item>
    <data:item no="31">1840</data:item>
    <data:item no="32">1832</data:item>
    <data:item no="33">1842</data:item>
    <data:item no="34">1833</data:item>
    <data:item no="35">1843</data:item>
    <data:item no="36">1836</data:item>
    <data:item no="37">1838</data:item>
    <data:item no="38">1843</data:item>
    <data:item no="39a">1834</data:item>
    <data:item no="39b">1834</data:item>
    <data:item no="39c">1835</data:item>
    <data:item no="40">1837</data:item>
    <data:item no="41">1838</data:item>
    <data:item no="42">1847</data:item>
    <data:item no="43">1839</data:item>
    <data:item no="44">1837</data:item>
    <data:item no="45">1831</data:item>
    <data:item no="46a">1846</data:item>
    <data:item no="46b">1847</data:item>
    <data:item no="47">1844</data:item>
    <data:item no="48">1840</data:item>
    <data:item no="49">1844</data:item>
    <data:item no="50">1838</data:item>
    <data:item no="51">1846</data:item>
    <data:item no="52">1856</data:item>
    <data:item no="53">1848</data:item>
    <data:item no="54">1842</data:item>
    <data:item no="55">1841</data:item>
    <data:item no="56">1854</data:item>
    <data:item no="57">1840</data:item>
    <data:item no="58">1829</data:item>
    <data:item no="59">1830</data:item>
    <data:item no="60">1855</data:item>
    <data:item no="61">1833</data:item>
    <data:item no="62">1841</data:item>
    <data:item no="63">1836</data:item>
    <data:item no="64">1831</data:item>
    <data:item no="65">1831</data:item>
    <data:item no="66">1835</data:item>
    <data:item no="67">1831</data:item>
    <data:item no="68">1837</data:item>
    <data:item no="69">1839</data:item>
    <data:item no="70">1834</data:item>
    <data:item no="71">1831</data:item>
    <data:item no="72">1830</data:item>
    <data:item no="73">1834</data:item>
    <data:item no="74">1831</data:item>
    <data:item no="75">1830</data:item>
    <data:item no="76">1834</data:item>
    <data:item no="77">1832</data:item>
    <data:item no="78">1831</data:item>
    <data:item no="79">1844</data:item>
    <data:item no="80">1831</data:item>
    <data:item no="81">1831</data:item>
    <data:item no="82">1832</data:item>
    <data:item no="83">1834</data:item>
    <data:item no="84">1829</data:item>
    <data:item no="85">1830</data:item>
    <data:item no="86a">1830</data:item>
    <data:item no="86b">1833</data:item>
    <data:item no="86c">1839</data:item>
  </data:dates>
  <xsl:variable name="no-date" select="document('')/*/data:dates/data:item"/>
  <xsl:variable name="bookdate" select="$no-date[@no = $bookno]"/>
  <xsl:variable name="bookid">
    <xsl:text>balzac</xsl:text>
    <xsl:value-of select="$bookdate"/>
    <xsl:text>FC</xsl:text>
    <xsl:value-of select="$bookno"/>
  </xsl:variable>
  <xsl:variable name="yamline">
    <xsl:text>---</xsl:text>
    <xsl:value-of select="$lf"/>
  </xsl:variable>
  <xsl:template match="/">
    <xsl:text>### </xsl:text>
    <xsl:variable name="chapcount"
      select="count(/tei:TEI/tei:text/tei:body//tei:div[@type='chapter'])"/>
    <xsl:value-of select="$filename"/>
    <xsl:text> chapters:</xsl:text>
    <xsl:value-of select="$chapcount"/>
    <xsl:choose>
      <xsl:when test="$chapcount = 1">???</xsl:when>
    </xsl:choose>
    <xsl:value-of select="$lf"/>
    <xsl:choose>
      <xsl:when test="$bookno = '86' or $bookno = '46' or $bookno = '39'"/>
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
          <xsl:value-of select="$bookdate"/>
          <xsl:value-of select="$lf"/>
          <xsl:text>modified: </xsl:text>
          <xsl:value-of select="$docdate"/>
          <xsl:value-of select="$lf"/>
          <xsl:text>title: </xsl:text>
          <xsl:value-of select="$doctitle"/>
          <xsl:value-of select="$lf"/>
          <xsl:value-of select="$yamline"/>
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
    <xsl:variable name="title">
      <xsl:apply-templates select="tei:head" mode="title"/>
    </xsl:variable>
    <xsl:variable name="meta">
      <xsl:text>identifier: </xsl:text>
      <xsl:value-of select="$docid"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>creator: </xsl:text>
      <xsl:value-of select="$byline"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>created: </xsl:text>
      <xsl:value-of select="$bookdate"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>modified: </xsl:text>
      <xsl:value-of select="$docdate"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>isPartOf: </xsl:text>
      <xsl:value-of select="$doctitle"/>
      <xsl:value-of select="$lf"/>
      <xsl:text>title: </xsl:text>
      <xsl:value-of select="$title"/>
      <xsl:value-of select="$lf"/>
    </xsl:variable>
    <xsl:value-of select="$yamline"/>
    <xsl:value-of select="$meta"/>
    <xsl:value-of select="$yamline"/>
  </xsl:template>
  <xsl:template match="tei:note" mode="md"/>
</xsl:transform>
